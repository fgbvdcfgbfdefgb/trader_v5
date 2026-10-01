"""
Offline market dataset: per-day episode sampling over BTC/ETH/LTC 1m bars.

Loads only the year-Parquet files it needs and caches a small LRU of them, so
memory stays flat even though the corpus is ~14M rows.
"""
from __future__ import annotations

import datetime as dt
import glob
import os
from collections import OrderedDict

import numpy as np
import pandas as pd

ASSETS = ["BTCUSDT", "ETHUSDT", "LTCUSDT"]
MIN_PER_DAY = 1440
WARMUP = 240          # minutes of context before the trading day starts

FEATURES = [
    "ret_1", "ret_5", "ret_15", "ret_60", "vol_30", "vol_120",
    "rsi_14", "macd", "bb_pos", "vol_z", "taker_ratio", "hl_range",
]
N_FEAT_PER_ASSET = len(FEATURES)
N_TIME_FEAT = 4       # sin/cos time-of-day, sin/cos day-of-week


def _rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    d = np.diff(close, prepend=close[0])
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    rd = pd.Series(dn).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    rs = ru / (rd + 1e-12)
    return (100.0 - 100.0 / (1.0 + rs)) / 100.0 - 0.5


def _ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()


def compute_features(df: pd.DataFrame) -> np.ndarray:
    c = df["close"].to_numpy(np.float64)
    h = df["high"].to_numpy(np.float64)
    lo = df["low"].to_numpy(np.float64)
    v = df["volume"].to_numpy(np.float64)
    tb = df["taker_buy_base"].to_numpy(np.float64)

    lc = np.log(np.clip(c, 1e-12, None))
    out = {}
    for n in (1, 5, 15, 60):
        out[f"ret_{n}"] = np.concatenate([np.zeros(n), lc[n:] - lc[:-n]]) * 100.0
    r1 = out["ret_1"]
    out["vol_30"] = pd.Series(r1).rolling(30, min_periods=1).std().fillna(0).to_numpy()
    out["vol_120"] = pd.Series(r1).rolling(120, min_periods=1).std().fillna(0).to_numpy()
    out["rsi_14"] = _rsi(c)
    macd = _ema(lc, 12) - _ema(lc, 26)
    out["macd"] = (macd - _ema(macd, 9)) * 100.0
    ma = pd.Series(c).rolling(60, min_periods=1).mean().to_numpy()
    sd = pd.Series(c).rolling(60, min_periods=1).std().fillna(0).to_numpy()
    out["bb_pos"] = np.clip((c - ma) / (2 * sd + 1e-9), -3, 3)
    lv = np.log1p(v)
    mv = pd.Series(lv).rolling(240, min_periods=1).mean().to_numpy()
    sv = pd.Series(lv).rolling(240, min_periods=1).std().fillna(0).to_numpy()
    out["vol_z"] = np.clip((lv - mv) / (sv + 1e-9), -5, 5)
    out["taker_ratio"] = np.where(v > 0, tb / (v + 1e-12), 0.5) - 0.5
    out["hl_range"] = (h - lo) / (c + 1e-12) * 100.0

    M = np.stack([out[k] for k in FEATURES], axis=1).astype(np.float32)
    return np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)


class MarketData:
    def __init__(self, root: str, assets=None, cache_years: int = 3):
        self.root = root
        self.assets = assets or ASSETS
        self._cache: OrderedDict = OrderedDict()
        self._cache_years = cache_years
        self._index = self._build_index()

    # ---------- index of tradable days ----------
    def _build_index(self):
        per = {}
        for a in self.assets:
            days = set()
            for f in sorted(glob.glob(os.path.join(self.root, a, "*.parquet"))):
                d = pd.read_parquet(f, columns=["open_time"])
                ts = pd.to_datetime(d["open_time"], unit="ms", utc=True)
                vc = ts.dt.strftime("%Y-%m-%d").value_counts()
                days |= set(vc[vc >= int(MIN_PER_DAY * 0.97)].index)
            per[a] = days
        common = set.intersection(*per.values())
        return sorted(common)

    @property
    def days(self):
        return self._index

    def _year(self, asset: str, year: int) -> pd.DataFrame:
        key = (asset, year)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        p = os.path.join(self.root, asset, f"{asset}-1m-{year}.parquet")
        if not os.path.exists(p):
            return pd.DataFrame()
        df = pd.read_parquet(p)
        df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.set_index("ts").sort_index()
        self._cache[key] = df
        while len(self._cache) > self._cache_years * len(self.assets):
            self._cache.popitem(last=False)
        return df

    def _slice(self, asset: str, start, end) -> pd.DataFrame:
        yrs = sorted({start.year, end.year})
        parts = [self._year(asset, y) for y in yrs]
        parts = [p for p in parts if len(p)]
        if not parts:
            return pd.DataFrame()
        df = pd.concat(parts) if len(parts) > 1 else parts[0]
        return df.loc[(df.index >= start) & (df.index < end)]

    # ---------- episode ----------
    def episode(self, day: str):
        """One trading day + WARMUP minutes of prior context, aligned across assets."""
        d0 = pd.Timestamp(day, tz="UTC")
        s = d0 - pd.Timedelta(minutes=WARMUP)
        e = d0 + pd.Timedelta(days=1)
        grid = pd.date_range(s, e, freq="1min", inclusive="left")

        feats, prices, raw = [], [], {}
        for a in self.assets:
            df = self._slice(a, s, e)
            if df.empty:
                return None
            df = df.reindex(grid).ffill().bfill()
            if df["close"].isna().any():
                return None
            raw[a] = df
            feats.append(compute_features(df))
            prices.append(df["close"].to_numpy(np.float32))

        X = np.concatenate(feats, axis=1)                      # [T, A*F]
        P = np.stack(prices, axis=1)                           # [T, A]

        tod = (grid.hour * 60 + grid.minute).to_numpy() / 1440.0
        dow = grid.dayofweek.to_numpy() / 7.0
        T = np.stack([np.sin(2*np.pi*tod), np.cos(2*np.pi*tod),
                      np.sin(2*np.pi*dow), np.cos(2*np.pi*dow)], 1).astype(np.float32)
        X = np.concatenate([X, T], axis=1).astype(np.float32)

        return {
            "day": day, "timestamps": grid, "features": X, "prices": P,
            "warmup": WARMUP, "assets": list(self.assets), "raw": raw,
        }

    def sample_day(self, rng: np.random.Generator, lo=None, hi=None) -> str:
        pool = self._index
        if lo or hi:
            pool = [d for d in pool if (not lo or d >= lo) and (not hi or d <= hi)]
        return pool[int(rng.integers(len(pool)))]


def forward_targets(prices: np.ndarray, horizons=(1, 5, 15, 60)) -> np.ndarray:
    """Supervised targets for the price predictor: future log-returns, in %."""
    lp = np.log(np.clip(prices, 1e-12, None))
    outs = []
    for h in horizons:
        f = np.concatenate([lp[h:], np.repeat(lp[-1:], h, axis=0)], axis=0)
        outs.append((f - lp) * 100.0)
    return np.stack(outs, axis=1).astype(np.float32)   # [T, H, A]
