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
            # tz-aware -> naive UTC datetime64[ns] so searchsorted is numeric.
            # (a tz-aware .to_numpy() yields an object array and breaks it)
            self._dt = (self.df["dt"].dt.tz_convert("UTC").dt.tz_localize(None)
                        .to_numpy(dtype="datetime64[ns]"))

    def window(self, end_ts, trail_days=TRAIL_DAYS, limit=120):
        """All articles in (end_ts - trail_days, end_ts]. Never future."""
        if not len(self.df):
            return self.df
        end_ts = pd.Timestamp(end_ts)
        if end_ts.tz is None:
            end_ts = end_ts.tz_localize("UTC")
        start = end_ts - pd.Timedelta(days=trail_days)
        a = np.datetime64(start.tz_convert("UTC").tz_localize(None), "ns")
        b = np.datetime64(end_ts.tz_convert("UTC").tz_localize(None), "ns")
        i0 = int(np.searchsorted(self._dt, a, "right"))
        i1 = int(np.searchsorted(self._dt, b, "right"))   # strict: no future
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
        self._tcache = None
        self._tdirty = 0

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
        lab = {int(i): str(l).lower() for i, l in mdl.config.id2label.items()}
        mf = os.path.join(d, "manifest.json")
        fb = json.load(open(mf)).get("labels") if os.path.exists(mf) else None
        self._loaded[name] = (tok, mdl, lab, fb)
        return self._loaded[name]

    POS_WORDS = ("positive", "bullish", "bull", "optimistic", "up")
    NEG_WORDS = ("negative", "bearish", "bear", "pessimistic", "down")
    NEU_WORDS = ("neutral", "none")

    @classmethod
    def _polarity(cls, labels, probs, fallback=None):
        """
        Map any 3-class sentiment head onto a single [-1, 1] score.

        Matches on whole sentiment words. If a model ships anonymous
        LABEL_0/1/2 names, fall back to the ordered label list recorded in
        manifest.json rather than guessing from digits.
        """
        def side(name):
            n = name.strip().lower()
            if any(w == n or w in n.split("_") or n.startswith(w)
                   for w in cls.NEU_WORDS):
                return 0
            if any(w == n or n.startswith(w) for w in cls.POS_WORDS):
                return 1
            if any(w == n or n.startswith(w) for w in cls.NEG_WORDS):
                return -1
            return None

        pos = neg = 0.0
        resolved = False
        for i, l in labels.items():
            i = int(i)
            if i >= len(probs):
                continue
            s = side(l)
            if s is None:
                continue
            resolved = True
            if s > 0:
                pos += probs[i]
            elif s < 0:
                neg += probs[i]

        if not resolved and fallback:
            for i, l in enumerate(fallback):
                if i >= len(probs):
                    break
                s = side(l)
                if s and s > 0:
                    pos += probs[i]
                elif s and s < 0:
                    neg += probs[i]
        return float(pos - neg)

    # ---- persistent per-headline score cache -------------------------------
    # Consecutive hours share a 7-day trailing window, so the same headline is
    # re-scored hundreds of times across an episode and across epochs. Scoring
    # each unique title once and remembering it is the single biggest speedup
    # available on a CPU-bound advisor.
    def _tcache_path(self):
        return os.path.join(self.cache_dir, "title_scores.pkl")

    def _load_tcache(self):
        if self._tcache is not None:
            return self._tcache
        import pickle
        try:
            with open(self._tcache_path(), "rb") as f:
                self._tcache = pickle.load(f)
        except Exception:
            self._tcache = {}
        return self._tcache

    def save_cache(self):
        if not self._tcache or not self._tdirty:
            return
        import pickle, tempfile
        try:
            d = os.path.dirname(self._tcache_path())
            with tempfile.NamedTemporaryFile("wb", dir=d, delete=False) as f:
                pickle.dump(self._tcache, f, protocol=4)
                tmp = f.name
            os.replace(tmp, self._tcache_path())
            self._tdirty = 0
        except Exception:
            pass

    @staticmethod
    def _tkey(t):
        return hashlib.md5(t.strip().lower().encode("utf8")).hexdigest()[:16]

    def score_texts(self, texts):
        if not texts or not self.enabled or not self.model_names:
            return np.zeros(len(texts), np.float32)
        cache = self._load_tcache()
        keys = [self._tkey(t) for t in texts]
        todo, seen = [], {}
        for t, k in zip(texts, keys):
            if k in cache or k in seen:
                continue
            seen[k] = True
            todo.append((k, t))

        if todo:
            import torch
            uniq = [t for _, t in todo]
            acc = np.zeros(len(uniq), np.float64)
            used = 0
            for name in self.model_names:
                try:
                    tok, mdl, lab, fb = self._load(name)
                except Exception:
                    continue
                out = []
                with torch.no_grad():
                    for i in range(0, len(uniq), self.batch):
                        b = uniq[i:i + self.batch]
                        enc = tok(b, padding=True, truncation=True,
                                  max_length=96, return_tensors="pt")
                        pr = torch.softmax(mdl(**enc).logits, -1).numpy()
                        out.extend(self._polarity(lab, p, fb) for p in pr)
                acc += np.asarray(out, np.float64)
                used += 1
            acc /= max(used, 1)
            for (k, _), v in zip(todo, acc):
                cache[k] = float(v)
            self._tdirty += len(todo)
            if self._tdirty >= 400:
                self.save_cache()

        return np.asarray([cache.get(k, 0.0) for k in keys], np.float32)

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
        self.save_cache()
        return out, notes
