#!/usr/bin/env python3
"""Tiny end-to-end check: 2 epochs, small nets. Proves the whole loop runs."""
import os, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
cmd = [sys.executable, os.path.join(ROOT, "src", "train.py"),
       "--epochs", "2", "--d-model", "96", "--n-layers", "2",
       "--inner-steps", "1", "--ppo-epochs", "2", "--png-every", "1",
       "--advisor-every", "240",
       "--outdir", os.path.join(ROOT, "outputs", "smoke"),
       "--ckpt", os.path.join(ROOT, "outputs", "smoke", "ckpt")] + sys.argv[1:]
print(" ".join(cmd)); sys.exit(subprocess.call(cmd))
