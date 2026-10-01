"""
CPU LLM news advisor.

Runs the chunk-assembled sentiment encoders (FinBERT / CryptoBERT /
fin-distilroberta) entirely on CPU and turns the news stream into a dense
feature vector the RL agents consume.

POINT-IN-TIME GUARANTEE
-----------------------
For a trading day D and wall-clock minute t inside that day, the advisor is
only ever shown articles with  datetime_utc <= t.  Nothing published later in
the day, and nothing from the future, can reach the model. A 7-day trailing
window supplies context. This is what keeps the backtest honest.
"""
from __future__ import annotations

import glob
import json
import os
import hashlib

import numpy as np
import pandas as pd

ASSET_KEY = {"BTCUSDT": "BTC", "ETHUSDT": "ETH", "LTCUSDT": "LTC"}
TRAIL_DAYS = 7
# per asset: [mean_sent, max_sent, min_sent, dispersion, log_volume, recency_w] + 2 market-wide
N_ADVISOR_FEAT = 3 * 6 + 2


class NewsStore:
    """Dated headline corpus, indexed for fast point-in-time slicing."""

    def __init__(self, root: str):
        fs = sorted(glob.glob(os.path.join(root, "news_*.parquet")))
        if not fs:
            self.df = pd.DataFrame(columns=["date", "datetime_utc", "asset",
                                            "title", "domain", "url"])
        else:
            self.df = pd.concat([pd.read_parquet(f) for f in fs],
                                ignore_index=True)
        if len(self.df):
            self.df["dt"] = pd.to_datetime(self.df["datetime_utc"], utc=True,
                                           errors="coerce")
            self.df = self.df.dropna(subset=["dt"]).sort_values("dt")
            self.df = self.df.drop_duplicates(subset=["url", "asset"])
            self.df = self.df.reset_index(drop=True)
            self._dt = self.df["dt"].to_numpy()

    def window(self, end_ts, trail_days=TRAIL_DAYS, limit=120):
        """All articles in (end_ts - trail_days, end_ts]. Never future."""
        if not len(self.df):
            return self.df
        start = end_ts - pd.Timedelta(days=trail_days)
        i0 = np.searchsorted(self._dt, np.datetime64(start.tz_convert("UTC").tz_localize(None)), "right")
        i1 = np.searchsorted(self._dt, np.datetime64(end_ts.tz_convert("UTC").tz_localize(None)), "right")
        sl = self.df.iloc[i0:i1]
        if len(sl) > limit:
            sl = sl.tail(limit)
        return sl


class LLMAdvisor:
    """Lazy-loaded CPU sentiment ensemble with an on-disk per-(day,slot) cache."""

    def __init__(self, models_root: str, news_root: str, cache_dir: str,
                 threads: int = 1, model_names=None, batch: int = 8,
                 enabled: bool = True):
        self.models_root = models_root
        self.cache_dir = cache_dir
        self.threads = threads
        self.batch = batch
        self.enabled = enabled
        os.makedirs(cache_dir, exist_ok=True)
        self.store = NewsStore(news_root)
        idx_p = os.path.join(models_root, "index.json")
        self.index = json.load(open(idx_p)) if os.path.exists(idx_p) else {}
        self.model_names = model_names or list(self.index.keys())
        self._loaded = {}

    # ---------- model plumbing ----------
    def _load(self, name):
        if name in self._loaded:
            return self._loaded[name]
        import torch
        from transformers import (AutoTokenizer,
                                  AutoModelForSequenceClassification)
        torch.set_num_threads(self.threads)
        d = os.path.join(self.models_root, name)
        tok = AutoTokenizer.from_pretrained(d, local_files_only=True)
        mdl = AutoModelForSequenceClassification.from_pretrained(
            d, local_files_only=True)
        mdl.eval()
        lab = {i: l.lower() for i, l in mdl.config.id2label.items()}
        self._loaded[name] = (tok, mdl, lab)
        return self._loaded[name]

    @staticmethod
    def _polarity(labels, probs):
        """Map any 3-class sentiment head onto a single [-1, 1] score."""
        pos = neg = 0.0
        for i, l in labels.items():
            if i >= len(probs):
                continue
            if any(k in l for k in ("pos", "bull", "1")) and "neg" not in l:
                pos += probs[i]
            elif any(k in l for k in ("neg", "bear", "0")) and "pos" not in l:
                neg += probs[i]
        return float(pos - neg)

    def score_texts(self, texts):
        if not texts or not self.enabled or not self.model_names:
            return np.zeros(len(texts), np.float32)
        import torch
        acc = np.zeros(len(texts), np.float64)
        used = 0
        for name in self.model_names:
            try:
                tok, mdl, lab = self._load(name)
            except Exception:
                continue
            out = []
            with torch.no_grad():
                for i in range(0, len(texts), self.batch):
                    b = texts[i:i + self.batch]
                    enc = tok(b, padding=True, truncation=True, max_length=96,
                              return_tensors="pt")
                    lg = mdl(**enc).logits
                    pr = torch.softmax(lg, -1).numpy()
                    out.extend(self._polarity(lab, p) for p in pr)
            acc += np.asarray(out, np.float64)
            used += 1
        return (acc / max(used, 1)).astype(np.float32)

    # ---------- the feature the agents actually see ----------
    def features_at(self, ts, assets):
        """Advisor vector visible at timestamp `ts` (point-in-time safe)."""
        key = hashlib.md5(
            f"{ts.floor('1h').isoformat()}|{','.join(self.model_names)}".encode()
        ).hexdigest()[:20]
        cpath = os.path.join(self.cache_dir, f"{key}.npz")
        if os.path.exists(cpath):
            try:
                z = np.load(cpath, allow_pickle=True)
                return z["vec"].astype(np.float32), str(z["summary"])
            except Exception:
                pass

        sl = self.store.window(ts)
        vec = np.zeros(N_ADVISOR_FEAT, np.float32)
        summary = "no news in window"
        if len(sl):
            titles = sl["title"].astype(str).tolist()
            scores = self.score_texts(titles)
            sl = sl.assign(score=scores)
            age_h = (ts - sl["dt"]).dt.total_seconds().to_numpy() / 3600.0
            w = np.exp(-age_h / 48.0)

            for ai, a in enumerate(assets):
                k = ASSET_KEY.get(a, a)
                m = sl["asset"].to_numpy() == k
                base = ai * 6
                if m.sum():
                    s = scores[m]
                    ww = w[m]
                    vec[base + 0] = float(np.average(s, weights=ww))
                    vec[base + 1] = float(s.max())
                    vec[base + 2] = float(s.min())
                    vec[base + 3] = float(s.std())
                    vec[base + 4] = float(np.log1p(m.sum()))
                    vec[base + 5] = float(ww.mean())
            mk = sl["asset"].to_numpy() == "MKT"
            if mk.sum():
                vec[18] = float(np.average(scores[mk], weights=w[mk]))
                vec[19] = float(np.log1p(mk.sum()))

            top = sl.assign(absw=np.abs(scores) * w).nlargest(3, "absw")
            summary = " | ".join(
                f"[{r.asset} {r.score:+.2f}] {str(r.title)[:90]}"
                for r in top.itertuples())

        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            np.savez_compressed(cpath, vec=vec, summary=summary)
        except Exception:
            pass
        return vec, summary

    def day_track(self, day, timestamps, assets, every=60):
        """
        Advisor features sampled through the day, forward-filled onto every
        minute. `every`=60 -> re-reads the news once an hour, which is both
        realistic and cheap on a busy CPU.
        """
        T = len(timestamps)
        out = np.zeros((T, N_ADVISOR_FEAT), np.float32)
        notes = []
        idxs = list(range(0, T, every))
        if idxs[-1] != T - 1:
            idxs.append(T - 1)
        prev = np.zeros(N_ADVISOR_FEAT, np.float32)
        last_i = 0
        for i in idxs:
            v, s = self.features_at(timestamps[i], assets)
            out[last_i:i + 1] = prev
            prev = v
            notes.append((str(timestamps[i]), s))
            last_i = i
        out[last_i:] = prev
        return out, notes
