"""
The three cooperating networks.

  PricePredictor  - causal TCN -> multi-horizon forward return + uncertainty
  MarketAnalyzer  - regime / volatility classifier that fuses news sentiment
  TraderPolicy    - PPO actor-critic over portfolio weights

The trader consumes the *latent* of the other two, not just their scalars, so
the three genuinely feed each other.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

HORIZONS = (1, 5, 15, 60)


class CausalConv(nn.Module):
    def __init__(self, c_in, c_out, k, d):
        super().__init__()
        self.pad = (k - 1) * d
        self.conv = nn.Conv1d(c_in, c_out, k, dilation=d)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class TCNBlock(nn.Module):
    def __init__(self, c, k, d, p=0.1):
        super().__init__()
        self.c1 = CausalConv(c, c, k, d)
        self.c2 = CausalConv(c, c, k, d)
        self.n1, self.n2 = nn.GroupNorm(1, c), nn.GroupNorm(1, c)
        self.do = nn.Dropout(p)

    def forward(self, x):
        h = self.do(F.gelu(self.n1(self.c1(x))))
        h = self.do(F.gelu(self.n2(self.c2(h))))
        return x + h


class Encoder(nn.Module):
    """Shared causal trunk: [B, T, F] -> [B, T, d]"""

    def __init__(self, n_feat, d, layers, k=3):
        super().__init__()
        self.inp = nn.Conv1d(n_feat, d, 1)
        self.blocks = nn.ModuleList([TCNBlock(d, k, 2 ** i) for i in range(layers)])
        self.out = nn.GroupNorm(1, d)

    def forward(self, x):
        h = self.inp(x.transpose(1, 2))
        for b in self.blocks:
            h = b(h)
        return self.out(h).transpose(1, 2)


class PricePredictor(nn.Module):
    """Predicts forward log-returns per asset per horizon, with aleatoric sigma."""

    def __init__(self, n_feat, n_assets, d=256, layers=4):
        super().__init__()
        self.enc = Encoder(n_feat, d, layers)
        self.n_assets, self.n_h = n_assets, len(HORIZONS)
        self.mu = nn.Linear(d, n_assets * self.n_h)
        self.logsig = nn.Linear(d, n_assets * self.n_h)
        self.d = d

    def forward(self, x):
        h = self.enc(x)
        B, T, _ = h.shape
        mu = self.mu(h).view(B, T, self.n_h, self.n_assets)
        ls = self.logsig(h).view(B, T, self.n_h, self.n_assets).clamp(-6, 3)
        return mu, ls, h

    def loss(self, x, y):
        """Gaussian NLL against realised forward returns y:[B,T,H,A]."""
        mu, ls, _ = self(x)
        inv = torch.exp(-2 * ls)
        return (0.5 * (inv * (y - mu) ** 2 + 2 * ls)).mean()


class MarketAnalyzer(nn.Module):
    """
    Reads price features + the advisor's news vector and emits:
      - regime logits (3): down / chop / up
      - predicted realised volatility
      - a latent the trader attends to
    """
    REGIMES = ["bear", "chop", "bull"]

    def __init__(self, n_feat, n_adv, n_assets, d=256, layers=3):
        super().__init__()
        self.enc = Encoder(n_feat, d, layers)
        self.news = nn.Sequential(nn.Linear(n_adv, d), nn.GELU(),
                                  nn.Linear(d, d))
        self.fuse = nn.Sequential(nn.Linear(2 * d, d), nn.GELU())
        self.regime = nn.Linear(d, n_assets * 3)
        self.vol = nn.Linear(d, n_assets)
        self.n_assets = n_assets
        self.d = d

    def forward(self, x, adv):
        h = self.enc(x)
        n = self.news(adv)
        z = self.fuse(torch.cat([h, n], -1))
        B, T, _ = z.shape
        return self.regime(z).view(B, T, self.n_assets, 3), self.vol(z), z

    def loss(self, x, adv, regime_y, vol_y):
        rl, vp, _ = self(x, adv)
        ce = F.cross_entropy(rl.reshape(-1, 3), regime_y.reshape(-1).long())
        mse = F.smooth_l1_loss(vp, vol_y)
        return ce + mse, ce.detach(), mse.detach()


class TraderPolicy(nn.Module):
    """
    PPO actor-critic. Action = portfolio weights over [BTC, ETH, LTC] in
    [-1,1]; the env renormalises so gross exposure <= max_gross.
    """

    def __init__(self, n_market, n_pred, n_reg, n_port, n_assets,
                 d=256, layers=2):
        super().__init__()
        n_in = n_market + n_pred + n_reg + n_port
        blocks = []
        cur = n_in
        for _ in range(layers):
            blocks += [nn.Linear(cur, d), nn.LayerNorm(d), nn.GELU()]
            cur = d
        self.trunk = nn.Sequential(*blocks)
        self.gru = nn.GRUCell(d, d)
        self.mu = nn.Linear(d, n_assets)
        self.logstd = nn.Parameter(torch.full((n_assets,), -0.9))
        self.v = nn.Linear(d, 1)
        self.d = d
        for m in (self.mu, self.v):
            nn.init.orthogonal_(m.weight, 0.01)
            nn.init.zeros_(m.bias)

    def forward(self, obs, hx=None):
        z = self.trunk(obs)
        hx = self.gru(z, hx if hx is not None else torch.zeros_like(z))
        mu = torch.tanh(self.mu(hx))
        std = self.logstd.exp().clamp(0.03, 1.0).expand_as(mu)
        return mu, std, self.v(hx).squeeze(-1), hx

    def act(self, obs, hx=None, deterministic=False):
        mu, std, v, hx = self(obs, hx)
        if deterministic:
            a = mu
            return a, torch.zeros(mu.shape[0], device=mu.device), v, hx
        dist = torch.distributions.Normal(mu, std)
        raw = dist.rsample()
        logp = dist.log_prob(raw).sum(-1)
        return raw.clamp(-1, 1), logp, v, hx

    def evaluate(self, obs, act, hx=None):
        mu, std, v, hx = self(obs, hx)
        dist = torch.distributions.Normal(mu, std)
        return (dist.log_prob(act).sum(-1), dist.entropy().sum(-1), v, hx)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
