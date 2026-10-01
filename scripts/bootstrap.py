#!/usr/bin/env python3
"""
Run this ONCE after cloning the repo into an offline workspace (Snowflake).
It rebuilds the chunked model weights, verifies the datasets, and prints the
hardware plan. It never touches the network.
"""
import os, sys, glob, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

print("=" * 68); print("trader_v5 bootstrap (offline)"); print("=" * 68)

print("\n[1/4] reassembling chunked model weights")
subprocess.check_call([sys.executable, os.path.join(ROOT, "models", "assemble.py")])

print("\n[2/4] verifying market data")
import pandas as pd
tot = 0
for sym in ("BTCUSDT", "ETHUSDT", "LTCUSDT"):
    fs = sorted(glob.glob(os.path.join(ROOT, "data", "market", sym, "*.parquet")))
    n = sum(len(pd.read_parquet(f, columns=["open_time"])) for f in fs)
    tot += n
    print(f"  {sym}: {len(fs)} files, {n:,} minute bars")
print(f"  total: {tot:,} bars")

print("\n[3/4] verifying news corpus")
fs = sorted(glob.glob(os.path.join(ROOT, "data", "news", "news_*.parquet")))
df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
print(f"  {len(df):,} articles over {df['date'].nunique():,} days "
      f"({df['date'].min()} -> {df['date'].max()})")

print("\n[4/4] hardware plan")
sys.path.insert(0, os.path.join(ROOT, "src"))
import resources
print(resources.describe(resources.detect()))
print("\nbootstrap OK - now run:  python src/train.py --epochs 2000")
