"""
One-day, three-asset trading environment.

Each epoch = one randomly chosen calendar day of BTC/ETH/LTC 1-minute bars.
The agent starts with START_CASH = $20 and is rewarded for compounding it
toward TARGET = $30 within the day.
"""
from __future__ import annotations

import numpy as np

START_CASH = 20.0
TARGET = 30.0


class TradingEnv:
    def __init__(self, episode, advisor_feats, decision_every=5,
                 fee=0.0004, slippage_bp=1.0, max_gross=1.0,
                 start_cash=START_CASH, target=TARGET,
                 dd_penalty=0.35, target_bonus=3.0):
        self.ep = episode
        self.adv = advisor_feats
        self.k = decision_every
        self.fee = fee
        self.slip = slippage_bp / 1e4
        self.max_gross = max_gross
        self.start_cash = start_cash
        self.target = target
        self.dd_penalty = dd_penalty
        self.target_bonus = target_bonus

        self.w0 = episode["warmup"]
        self.P = episode["prices"]
        self.X = episode["features"]
        self.T = len(self.P)
        self.n_assets = self.P.shape[1]
        self.steps = list(range(self.w0, self.T - 1, self.k))
        self.reset()

    # ---------- core ----------
    def reset(self):
        self.i = 0
        self.t = self.steps[0]
        self.equity = self.start_cash
        self.peak = self.start_cash
        self.w = np.zeros(self.n_assets, np.float32)
        self.hist = {"t": [], "equity": [], "weights": [], "reward": [],
                     "price": [], "turnover": [], "cost": [], "drawdown": []}
        return self._obs_parts()

    def _obs_parts(self):
        """Market window + portfolio state. Agent nets add their own latents."""
        return {
            "t": self.t,
            "market": self.X[self.t],
            "advisor": self.adv[self.t],
            "portfolio": np.array(
                [*self.w,
                 self.equity / self.start_cash - 1.0,
                 (self.equity - self.peak) / self.peak,
                 self.i / max(len(self.steps) - 1, 1),
                 np.clip((self.target - self.equity) / self.start_cash, -2, 2)],
                np.float32),
        }

    def step(self, action: np.ndarray):
        a = np.clip(np.asarray(action, np.float32), -1, 1)
        gross = np.abs(a).sum()
        if gross > self.max_gross:
            a = a * (self.max_gross / (gross + 1e-9))

        turnover = float(np.abs(a - self.w).sum())
        cost = turnover * (self.fee + self.slip) * self.equity
        self.w = a

        t0, t1 = self.t, min(self.t + self.k, self.T - 1)
        r_assets = (self.P[t1] / np.clip(self.P[t0], 1e-12, None)) - 1.0
        pnl = float(self.equity * np.dot(self.w, r_assets))

        prev_eq = self.equity
        self.equity = max(self.equity + pnl - cost, 0.01)
        self.peak = max(self.peak, self.equity)
        dd = (self.equity - self.peak) / self.peak

        # log-growth reward, scaled, minus drawdown pressure
        reward = float(np.log(self.equity / prev_eq) * 100.0)
        reward += self.dd_penalty * dd * 10.0
        reward -= 0.02 * turnover

        self.hist["t"].append(t0)
        self.hist["equity"].append(self.equity)
        self.hist["weights"].append(self.w.copy())
        self.hist["reward"].append(reward)
        self.hist["price"].append(self.P[t0].copy())
        self.hist["turnover"].append(turnover)
        self.hist["cost"].append(cost)
        self.hist["drawdown"].append(dd)

        self.i += 1
        done = self.i >= len(self.steps) or self.equity <= 0.25 * self.start_cash
        if not done:
            self.t = self.steps[self.i]

        if done:
            # terminal shaping toward the $30 objective
            reached = self.equity >= self.target
            reward += self.target_bonus * (1.0 if reached else 0.0)
            reward += 2.0 * np.clip(
                (self.equity - self.start_cash) / (self.target - self.start_cash),
                -1.0, 1.5)

        return self._obs_parts(), reward, done, self._info()

    def _info(self):
        eq = np.asarray(self.hist["equity"], np.float64)
        if len(eq) < 2:
            return {"equity": self.equity, "return_pct": 0.0, "sharpe": 0.0,
                    "max_dd": 0.0, "hit_target": False, "turnover": 0.0}
        r = np.diff(np.log(eq))
        sharpe = float(r.mean() / (r.std() + 1e-12) * np.sqrt(len(r))) if len(r) > 2 else 0.0
        return {
            "equity": float(self.equity),
            "return_pct": float((self.equity / self.start_cash - 1) * 100),
            "sharpe": sharpe,
            "max_dd": float(min(self.hist["drawdown"])),
            "hit_target": bool(self.equity >= self.target),
            "turnover": float(np.sum(self.hist["turnover"])),
            "cost": float(np.sum(self.hist["cost"])),
        }

    @property
    def n_steps(self):
        return len(self.steps)


def regime_labels(prices: np.ndarray, lookfwd=60, thresh=0.0015):
    """Weak labels for the analyzer: future drift bucketed into bear/chop/bull."""
    lp = np.log(np.clip(prices, 1e-12, None))
    fwd = np.concatenate([lp[lookfwd:], np.repeat(lp[-1:], lookfwd, 0)], 0) - lp
    y = np.ones_like(fwd, dtype=np.int64)
    y[fwd > thresh] = 2
    y[fwd < -thresh] = 0
    return y


def realised_vol(prices: np.ndarray, win=60):
    import pandas as pd
    lp = np.log(np.clip(prices, 1e-12, None))
    r = np.diff(lp, axis=0, prepend=lp[:1])
    return (pd.DataFrame(r).rolling(win, min_periods=1).std()
              .fillna(0).to_numpy() * 100.0).astype(np.float32)
