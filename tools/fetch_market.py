#!/usr/bin/env python3
"""
Download full-history 1-minute klines for BTC/ETH/LTC from Binance public data
dumps, normalise them, and write ONE Parquet file per symbol per year.

Designed for tiny disks: each monthly .zip is downloaded, parsed, and deleted
immediately. Peak extra disk use is one month (~2 MB).
"""
import io
import os
import sys
import time
import zipfile
import datetime as dt

import numpy as np
import pandas as pd
import requests

BASE = "https://data.binance.vision/data/spot/monthly/klines/{sym}/1m/{sym}-1m-{ym}.zip"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "LTCUSDT"]
START = dt.date(2017, 8, 1)
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "market")

COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]
NUM = ["open", "high", "low", "close", "volume", "quote_volume",
       "trades", "taker_buy_base", "taker_buy_quote"]


def months(start: dt.date, end: dt.date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m == 13:
            y, m = y + 1, 1


def to_ms(series: pd.Series) -> pd.Series:
    """Binance switched epoch units to microseconds in 2025. Auto-detect."""
    s = pd.to_numeric(series, errors="coerce")
    med = s.dropna().median()
    if med > 1e15:          # microseconds
        s = s // 1000
    elif med > 1e14:        # defensive
        s = s // 1000
    return s.astype("int64")


def fetch_month(sym: str, ym: str, session: requests.Session):
    url = BASE.format(sym=sym, ym=ym)
    for attempt in range(4):
        try:
            r = session.get(url, timeout=90)
        except Exception as e:
            time.sleep(2 + attempt * 3)
            continue
        if r.status_code == 404:
            return None
        if r.status_code != 200:
            time.sleep(2 + attempt * 3)
            continue
        try:
            zf = zipfile.ZipFile(io.BytesIO(r.content))
        except zipfile.BadZipFile:
            time.sleep(2)
            continue
        name = zf.namelist()[0]
        with zf.open(name) as fh:
            head = fh.read(64)
        has_header = b"open_time" in head
        with zf.open(name) as fh:
            df = pd.read_csv(
                fh,
                header=0 if has_header else None,
                names=None if has_header else COLS,
            )
        if has_header:
            df.columns = COLS[: len(df.columns)]
        del r, zf
        return df
    return None


def main():
    os.makedirs(OUT, exist_ok=True)
    today = dt.date.today()
    end = (today.replace(day=1) - dt.timedelta(days=1))  # last complete month
    session = requests.Session()

    for sym in SYMBOLS:
        sym_dir = os.path.join(OUT, sym)
        os.makedirs(sym_dir, exist_ok=True)
        buf, cur_year, got_any = [], None, False

        def flush(year):
            if not buf:
                return
            d = pd.concat(buf, ignore_index=True)
            d = d.drop_duplicates(subset="open_time").sort_values("open_time")
            d = d.reset_index(drop=True)
            p = os.path.join(sym_dir, f"{sym}-1m-{year}.parquet")
            d.to_parquet(p, compression="zstd", index=False)
            mb = os.path.getsize(p) / 1e6
            print(f"  [write] {os.path.basename(p)}  rows={len(d):>7}  {mb:.1f} MB",
                  flush=True)
            buf.clear()

        print(f"=== {sym} ===", flush=True)
        for ym in months(START, end):
            year = int(ym[:4])
            if cur_year is not None and year != cur_year:
                flush(cur_year)
            cur_year = year

            df = fetch_month(sym, ym, session)
            if df is None:
                if got_any:
                    print(f"  [miss ] {ym}", flush=True)
                continue
            got_any = True

            df["open_time"] = to_ms(df["open_time"])
            df["close_time"] = to_ms(df["close_time"])
            for c in NUM:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.drop(columns=["ignore"], errors="ignore")
            df = df.dropna(subset=["open", "high", "low", "close"])
            df = df[(df["open"] > 0) & (df["close"] > 0)]
            df["trades"] = df["trades"].fillna(0).astype("int64")
            for c in ["open", "high", "low", "close", "volume",
                      "quote_volume", "taker_buy_base", "taker_buy_quote"]:
                df[c] = df[c].astype("float64")
            buf.append(df)
            print(f"  [ok   ] {ym}  rows={len(df)}", flush=True)
        flush(cur_year)

    print("MARKET_DONE", flush=True)


if __name__ == "__main__":
    main()
