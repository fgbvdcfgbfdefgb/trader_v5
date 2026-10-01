#!/usr/bin/env python3
"""
Download the CPU sentiment-encoder bundle from Hugging Face and split every
file >90 MB into numbered chunks so the whole bundle can live in plain git
(no LFS, no 100 MB file-limit violations).

On the offline side, models/assemble.py rebuilds the originals.
"""
import hashlib
import json
import os
import sys

import requests

HF = "https://huggingface.co/{repo}/resolve/main/{fn}"
ROOT = os.path.join(os.path.dirname(__file__), "..", "models")
CHUNK = 90 * 1024 * 1024

MODELS = {
    # financial-domain sentiment, 110M params - the workhorse
    "finbert": {
        "repo": "ProsusAI/finbert",
        "files": ["config.json", "vocab.txt", "tokenizer_config.json",
                  "special_tokens_map.json", "pytorch_model.bin"],
        "task": "sentiment",
        "labels": ["positive", "negative", "neutral"],
    },
    # crypto-social-media domain, BERT-base
    "cryptobert": {
        "repo": "ElKulako/cryptobert",
        "files": ["config.json", "vocab.json", "merges.txt",
                  "tokenizer_config.json", "special_tokens_map.json",
                  "pytorch_model.bin"],
        "task": "sentiment",
        "labels": ["bearish", "neutral", "bullish"],
    },
    # distilled financial-news sentiment, fast fallback
    "fin_distilroberta": {
        "repo": "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
        "files": ["config.json", "vocab.json", "merges.txt",
                  "tokenizer_config.json", "special_tokens_map.json",
                  "pytorch_model.bin"],
        "task": "sentiment",
        "labels": ["negative", "neutral", "positive"],
    },
}


def get(url, dest, session):
    with session.get(url, stream=True, timeout=180) as r:
        if r.status_code != 200:
            return False
        tmp = dest + ".part"
        with open(tmp, "wb") as f:
            for c in r.iter_content(1 << 20):
                f.write(c)
        os.replace(tmp, dest)
    return True


def split(path, manifest):
    size = os.path.getsize(path)
    rel = os.path.basename(path)
    if size <= CHUNK:
        manifest["files"][rel] = {"chunks": 0, "size": size,
                                  "sha256": sha(path)}
        return
    digest = sha(path)
    n = 0
    with open(path, "rb") as f:
        while True:
            buf = f.read(CHUNK)
            if not buf:
                break
            with open(f"{path}.part{n:03d}", "wb") as o:
                o.write(buf)
            n += 1
    os.remove(path)
    manifest["files"][rel] = {"chunks": n, "size": size, "sha256": digest}
    print(f"    split into {n} chunks ({size/1e6:.0f} MB)", flush=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    session = requests.Session()
    os.makedirs(ROOT, exist_ok=True)
    index = {}
    for name, spec in MODELS.items():
        d = os.path.join(ROOT, name)
        os.makedirs(d, exist_ok=True)
        manifest = {"repo": spec["repo"], "task": spec["task"],
                    "labels": spec["labels"], "files": {}}
        print(f"=== {name}  ({spec['repo']}) ===", flush=True)
        for fn in spec["files"]:
            dest = os.path.join(d, fn)
            ok = get(HF.format(repo=spec["repo"], fn=fn), dest, session)
            if not ok:
                print(f"    [skip] {fn}", flush=True)
                continue
            mb = os.path.getsize(dest) / 1e6
            print(f"    [ok  ] {fn}  {mb:.1f} MB", flush=True)
            split(dest, manifest)
        with open(os.path.join(d, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        index[name] = {"repo": spec["repo"], "task": spec["task"],
                       "labels": spec["labels"]}
    with open(os.path.join(ROOT, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print("MODELS_DONE", flush=True)


if __name__ == "__main__":
    main()
