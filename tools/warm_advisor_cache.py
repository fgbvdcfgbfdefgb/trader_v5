#!/usr/bin/env python3
"""
Optional: pre-compute the LLM advisor cache for a date range using several CPU
workers. Entirely offline. Run it once and every epoch that touches a warmed
day costs ~0s of advisor time instead of ~45s.

    python tools/warm_advisor_cache.py --workers 4 --lo 2023-01-01 --hi 2024-01-01
"""
import argparse, os, sys, time
import multiprocessing as mp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def worker(args):
    days, cache_dir, every, threads = args
    import pandas as pd
    from advisor import LLMAdvisor
    from dataset import ASSETS, WARMUP
    a = LLMAdvisor(os.path.join(ROOT, "models"),
                   os.path.join(ROOT, "data", "news"), cache_dir,
                   threads=threads, batch=16)
    done = 0
    for d in days:
        d0 = pd.Timestamp(d, tz="UTC") - pd.Timedelta(minutes=WARMUP)
        grid = pd.date_range(d0, d0 + pd.Timedelta(minutes=1680), freq="1min")[:1680]
        a.day_track(d, grid, ASSETS, every=every)
        done += 1
        if done % 10 == 0:
            print(f"  [{os.getpid()}] {done}/{len(days)}", flush=True)
    a.save_cache()
    return done


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--every", type=int, default=60)
    p.add_argument("--lo", default="")
    p.add_argument("--hi", default="")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--cache", default=os.path.join(ROOT, "outputs", "advisor_cache"))
    a = p.parse_args()

    from dataset import MarketData, ASSETS, days_with_news
    md = MarketData(os.path.join(ROOT, "data", "market"), ASSETS)
    days = md.days
    if a.lo:
        days = [d for d in days if d >= a.lo]
    if a.hi:
        days = [d for d in days if d <= a.hi]
    days = days_with_news(days, os.path.join(ROOT, "data", "news"), 3)
    print(f"warming {len(days)} days with {a.workers} workers -> {a.cache}")
    os.makedirs(a.cache, exist_ok=True)

    # NOTE: each worker gets its own cache dir to avoid clobbering the shared
    # pickle, then we merge at the end.
    chunks = [days[i::a.workers] for i in range(a.workers)]
    jobs = [(c, os.path.join(a.cache, f"w{i}"), a.every, a.threads)
            for i, c in enumerate(chunks)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        pool.map(worker, jobs)

    # merge per-worker title caches + npz files into the shared dir
    import pickle, shutil, glob
    merged = {}
    base = os.path.join(a.cache, "title_scores.pkl")
    if os.path.exists(base):
        merged.update(pickle.load(open(base, "rb")))
    for i in range(a.workers):
        wd = os.path.join(a.cache, f"w{i}")
        tp = os.path.join(wd, "title_scores.pkl")
        if os.path.exists(tp):
            merged.update(pickle.load(open(tp, "rb")))
        for f in glob.glob(os.path.join(wd, "*.npz")):
            shutil.move(f, os.path.join(a.cache, os.path.basename(f)))
        shutil.rmtree(wd, ignore_errors=True)
    pickle.dump(merged, open(base, "wb"), protocol=4)
    print(f"done in {(time.time()-t0)/60:.1f} min; {len(merged):,} titles cached")


if __name__ == "__main__":
    main()
