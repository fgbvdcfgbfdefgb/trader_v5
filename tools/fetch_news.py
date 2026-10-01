#!/usr/bin/env python3
"""
Build the dated crypto-news corpus for BTC / ETH / LTC.

Sources (all free, no API key):
  1. HF  edaschau/bitcoin_news          - dated BTC headlines + article text, 2011-2025
  2. HF  SahandNZ/cryptonews-articles   - dated cryptonews.com articles, 2021-2023
  3. Cointelegraph sitemaps             - ~100k dated article URLs incl. 2026,
                                          headline recovered from the slug
  4. GDELT DOC API (optional, --gdelt)  - opportunistic top-up, heavily rate limited

Output: data/news/news_<YEAR>.parquet, plus daily_index.parquet.
Every row carries an exact UTC timestamp so the advisor can slice
point-in-time without leaking the future.
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys
import time
import datetime as dt

import numpy as np
import pandas as pd
import requests

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "news")
UA = {"User-Agent": "Mozilla/5.0 (compatible; trader_v5-research/1.0)"}

KEYS = {
    "BTC": re.compile(r"\b(bitcoin|btc|satoshi|halving)\b", re.I),
    "ETH": re.compile(r"\b(ethereum|ether|eth|vitalik|erc-?20|merge|staking)\b", re.I),
    "LTC": re.compile(r"\b(litecoin|ltc)\b", re.I),
}
MKT = re.compile(
    r"\b(crypto|cryptocurrency|altcoin|blockchain|sec|etf|regulat|exchange|"
    r"binance|coinbase|defi|stablecoin|rally|crash|selloff|surge|plunge|bull|bear)\b",
    re.I)


def tag_assets(text: str):
    t = text or ""
    hits = [k for k, rx in KEYS.items() if rx.search(t)]
    if not hits and MKT.search(t):
        hits = ["MKT"]
    return hits


def rows_from(df, title_col, time_col, url_col, source):
    out = []
    for r in df.itertuples(index=False):
        title = str(getattr(r, title_col, "") or "").strip()
        if len(title) < 12:
            continue
        ts = getattr(r, time_col, None)
        if ts is None or (isinstance(ts, float) and pd.isna(ts)):
            continue
        try:
            t = pd.to_datetime(ts, utc=True, errors="coerce")
        except Exception:
            continue
        if pd.isna(t):
            continue
        url = str(getattr(r, url_col, "") or "")
        for a in tag_assets(title):
            out.append({"date": t.strftime("%Y-%m-%d"),
                        "datetime_utc": t.strftime("%Y-%m-%d %H:%M:%S"),
                        "asset": a, "title": title[:300],
                        "domain": source, "url": url[:300], "language": "en"})
    return out


# ---------------------------------------------------------------- source 1/2
def hf_csv(repo, fn, session):
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{fn}"
    r = session.get(url, timeout=300, headers=UA)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content), on_bad_lines="skip",
                       engine="python")


def src_bitcoin_news(session):
    rows = []
    for fn in ("BTC_match_title.csv", "BTC_match_text.csv"):
        try:
            df = hf_csv("edaschau/bitcoin_news", fn, session)
        except Exception as e:
            print(f"  [warn] {fn}: {e}", flush=True)
            continue
        df = df[[c for c in ("date_time", "title", "url", "source") if c in df]]
        rows += rows_from(df, "title", "date_time", "url", "yahoo/news")
        print(f"  {fn}: {len(df)} raw rows", flush=True)
        del df
    return rows


def src_cryptonews(session):
    rows = []
    for fn in ("train.csv", "validation.csv", "test.csv"):
        try:
            df = hf_csv("SahandNZ/cryptonews-articles-with-price-momentum-labels",
                        fn, session)
        except Exception as e:
            print(f"  [warn] {fn}: {e}", flush=True)
            continue
        rows += rows_from(df, "text", "datetime", "url", "cryptonews.com")
        print(f"  {fn}: {len(df)} raw rows", flush=True)
        del df
    return rows


# ------------------------------------------------------------------ source 3
SLUG_RX = re.compile(r"https?://cointelegraph\.com/(?:news|press-releases|"
                     r"innovation-circle|explained)/([a-z0-9\-]+)")


def src_cointelegraph(session, max_maps=6):
    rows = []
    for i in range(1, max_maps + 1):
        url = f"https://cointelegraph.com/sitemap/articles/{i}.xml"
        try:
            r = session.get(url, timeout=180, headers=UA)
        except Exception:
            break
        if r.status_code != 200 or b"<loc>" not in r.content:
            break
        txt = r.text
        locs = re.findall(r"<loc>(.*?)</loc>", txt)
        mods = re.findall(r"<lastmod>(.*?)</lastmod>", txt)
        n = 0
        for loc, mod in zip(locs, mods):
            m = SLUG_RX.match(loc)
            if not m:
                continue
            title = m.group(1).replace("-", " ").strip()
            if len(title) < 12:
                continue
            t = pd.to_datetime(mod, utc=True, errors="coerce")
            if pd.isna(t):
                continue
            for a in tag_assets(title):
                rows.append({"date": t.strftime("%Y-%m-%d"),
                             "datetime_utc": t.strftime("%Y-%m-%d %H:%M:%S"),
                             "asset": a, "title": title[:300],
                             "domain": "cointelegraph.com",
                             "url": loc[:300], "language": "en"})
                n += 1
        print(f"  sitemap {i}: {len(locs)} urls -> {n} tagged rows", flush=True)
        del r, txt, locs, mods
    return rows


# ------------------------------------------------------------------ source 4
WP_SITES = {
    "cryptoslate.com": "https://cryptoslate.com/sitemap_index.xml",
    "newsbtc.com": "https://www.newsbtc.com/sitemap_index.xml",
    "bitcoinist.com": "https://bitcoinist.com/sitemap_index.xml",
    "ambcrypto.com": "https://ambcrypto.com/sitemap_index.xml",
}
WP_SLUG = re.compile(r"https?://[^/]+/(?:news/|\d{4}/\d{2}/\d{2}/)?([a-z0-9\-]{12,})/?$")


def src_wordpress(session, max_maps_per_site=80):
    """
    WordPress post-sitemaps carry a per-article <lastmod> that, for news
    outlets, tracks publication closely. Headline is recovered from the slug.
    """
    rows = []
    for domain, idx in WP_SITES.items():
        try:
            r = session.get(idx, timeout=90, headers=UA)
            maps = [m for m in re.findall(r"<loc>(.*?)</loc>", r.text)
                    if "post-sitemap" in m]
        except Exception as e:
            print(f"  [warn] {domain}: {e}", flush=True)
            continue
        maps = maps[-max_maps_per_site:]          # newest chunks first
        got = 0
        for u in maps:
            try:
                rr = session.get(u, timeout=90, headers=UA)
            except Exception:
                continue
            locs = re.findall(r"<loc>(.*?)</loc>", rr.text)
            mods = re.findall(r"<lastmod>(.*?)</lastmod>", rr.text)
            if len(locs) != len(mods):
                continue
            for loc, mod in zip(locs, mods):
                m = WP_SLUG.match(loc)
                if not m:
                    continue
                title = m.group(1).replace("-", " ").strip()
                if len(title) < 12:
                    continue
                t = pd.to_datetime(mod, utc=True, errors="coerce")
                if pd.isna(t):
                    continue
                for a in tag_assets(title):
                    rows.append({"date": t.strftime("%Y-%m-%d"),
                                 "datetime_utc": t.strftime("%Y-%m-%d %H:%M:%S"),
                                 "asset": a, "title": title[:300],
                                 "domain": domain, "url": loc[:300],
                                 "language": "en"})
                    got += 1
            del rr
        print(f"  {domain}: {len(maps)} sitemaps -> {got} tagged rows", flush=True)
    return rows


def despike(df, max_per_domain_day=250):
    """
    Sitemap <lastmod> is a *modification* time. When a site bulk-re-touches its
    archive, thousands of old articles collapse onto one date. Those rows carry
    a false timestamp and would inject phantom news, so drop any
    (domain, date) bucket that is implausibly large.
    """
    g = df.groupby(["domain", "date"]).size()
    bad = set(g[g > max_per_domain_day].index)
    if not bad:
        return df, 0
    key = list(zip(df["domain"], df["date"]))
    mask = np.array([k not in bad for k in key])
    return df[mask].reset_index(drop=True), int((~mask).sum())


def src_gdelt(session, start_year, sleep=12.0):
    Q = {"BTC": "(bitcoin OR BTC) (crypto OR price OR market)",
         "ETH": "(ethereum OR ETH) (crypto OR price OR market)",
         "LTC": "(litecoin OR LTC) (crypto OR price OR market)"}
    rows, today = [], dt.date.today()
    y, m = start_year, 1
    while dt.date(y, m, 1) <= today:
        a = dt.date(y, m, 1)
        y2, m2 = (y, m + 1) if m < 12 else (y + 1, 1)
        for asset, q in Q.items():
            try:
                r = session.get(
                    "https://api.gdeltproject.org/api/v2/doc/doc",
                    params={"query": q, "mode": "artlist", "maxrecords": 250,
                            "format": "json", "sort": "hybridrel",
                            "startdatetime": a.strftime("%Y%m%d") + "000000",
                            "enddatetime": dt.date(y2, m2, 1).strftime("%Y%m%d") + "000000"},
                    timeout=90, headers=UA)
                if r.text.strip().startswith("{"):
                    for art in r.json().get("articles", []):
                        t = pd.to_datetime(art.get("seendate"), utc=True,
                                           format="%Y%m%dT%H%M%SZ", errors="coerce")
                        if pd.isna(t):
                            continue
                        rows.append({"date": t.strftime("%Y-%m-%d"),
                                     "datetime_utc": t.strftime("%Y-%m-%d %H:%M:%S"),
                                     "asset": asset,
                                     "title": (art.get("title") or "")[:300],
                                     "domain": art.get("domain", ""),
                                     "url": (art.get("url") or "")[:300],
                                     "language": art.get("language", "")})
            except Exception:
                pass
            time.sleep(sleep)
        print(f"  gdelt {a:%Y-%m} total={len(rows)}", flush=True)
        y, m = y2, m2
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gdelt", action="store_true")
    ap.add_argument("--gdelt-start", type=int, default=2021)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    s = requests.Session()
    rows = []

    print("[1/4] HF edaschau/bitcoin_news", flush=True)
    rows += src_bitcoin_news(s)
    print(f"  -> {len(rows)} rows", flush=True)

    print("[2/4] HF SahandNZ/cryptonews", flush=True)
    rows += src_cryptonews(s)
    print(f"  -> {len(rows)} rows", flush=True)

    print("[3/4] Cointelegraph sitemaps", flush=True)
    rows += src_cointelegraph(s)
    print(f"  -> {len(rows)} rows", flush=True)

    print("[4/4] WordPress crypto outlets", flush=True)
    rows += src_wordpress(s)
    print(f"  -> {len(rows)} rows", flush=True)

    if args.gdelt:
        print("[+] GDELT top-up", flush=True)
        rows += src_gdelt(s, args.gdelt_start)

    df = pd.DataFrame(rows)
    if df.empty:
        print("NO NEWS COLLECTED"); return
    df = df.drop_duplicates(subset=["url", "asset", "title"])
    df, nd = despike(df)
    print(f"  de-spiked {nd:,} rows from bulk sitemap re-touches", flush=True)
    df = df.sort_values("datetime_utc").reset_index(drop=True)

    # merge anything already on disk (resume-friendly)
    import glob
    prev = []
    for f in glob.glob(os.path.join(OUT, "news_*.parquet")):
        try:
            prev.append(pd.read_parquet(f))
        except Exception:
            pass
    if prev:
        df = pd.concat([pd.concat(prev, ignore_index=True), df], ignore_index=True)
        df = df.drop_duplicates(subset=["url", "asset", "title"])
        df = df.sort_values("datetime_utc").reset_index(drop=True)

    df["year"] = df["date"].str[:4]
    for yr, g in df.groupby("year"):
        g.drop(columns=["year"]).to_parquet(
            os.path.join(OUT, f"news_{yr}.parquet"), compression="zstd",
            index=False)
    idx = df.groupby(["date", "asset"]).size().rename("n_articles").reset_index()
    idx.to_parquet(os.path.join(OUT, "daily_index.parquet"),
                   compression="zstd", index=False)

    print("\n" + "=" * 60)
    print(f"articles      : {len(df):,}")
    print(f"distinct days : {df['date'].nunique():,}")
    print(f"date range    : {df['date'].min()} -> {df['date'].max()}")
    print(df["asset"].value_counts().to_string())
    print("per-year rows:")
    print(df["year"].value_counts().sort_index().to_string())
    print("NEWS_DONE", flush=True)


if __name__ == "__main__":
    main()
