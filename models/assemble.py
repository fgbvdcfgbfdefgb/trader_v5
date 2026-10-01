#!/usr/bin/env python3
"""
Rebuild chunked model weights in-place. Safe to run repeatedly and safe to run
offline - it only touches files already present in the cloned repo.

    python models/assemble.py
"""
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def assemble_model(d, verify=True):
    mpath = os.path.join(d, "manifest.json")
    if not os.path.exists(mpath):
        return
    man = json.load(open(mpath))
    for fn, meta in man["files"].items():
        target = os.path.join(d, fn)
        n = meta.get("chunks", 0)
        if n == 0:
            continue
        if os.path.exists(target) and os.path.getsize(target) == meta["size"]:
            continue
        parts = [f"{target}.part{i:03d}" for i in range(n)]
        missing = [p for p in parts if not os.path.exists(p)]
        if missing:
            print(f"  !! {fn}: missing {len(missing)} chunk(s) - skipped")
            continue
        with open(target + ".tmp", "wb") as out:
            for p in parts:
                with open(p, "rb") as f:
                    while True:
                        b = f.read(1 << 20)
                        if not b:
                            break
                        out.write(b)
        os.replace(target + ".tmp", target)
        ok = (not verify) or sha(target) == meta["sha256"]
        print(f"  {'OK ' if ok else 'BAD'} {fn}  "
              f"({os.path.getsize(target)/1e6:.0f} MB from {n} chunks)")
        if not ok:
            raise SystemExit(f"checksum mismatch for {target}")


def main():
    idx = os.path.join(ROOT, "index.json")
    names = list(json.load(open(idx)).keys()) if os.path.exists(idx) else [
        x for x in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, x))]
    for name in names:
        d = os.path.join(ROOT, name)
        if not os.path.isdir(d):
            continue
        print(f"[assemble] {name}")
        assemble_model(d)
    print("assemble: done")


if __name__ == "__main__":
    main()
