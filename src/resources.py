"""
Hardware discovery + automatic capacity planning.

Everything downstream (model widths, batch sizes, device placement, how many
agent processes run in parallel) is derived from what this module finds, so the
exact same repo runs on 4xT4, on 1 small GPU, or on pure CPU.
"""
from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass, asdict, field

import torch


@dataclass
class Plan:
    n_gpus: int
    gpu_names: list = field(default_factory=list)
    gpu_mem_gb: list = field(default_factory=list)
    free_mem_gb: list = field(default_factory=list)
    cpu_count: int = 1
    ram_gb: float = 0.0
    size_tier: str = "small"
    # placement
    dev_predictor: str = "cpu"
    dev_analyzer: str = "cpu"
    dev_trader: str = "cpu"
    dev_advisor: str = "cpu"
    # capacity
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    batch_size: int = 64
    advisor_batch: int = 8
    advisor_threads: int = 1
    parallel_mode: str = "multiprocess"
    amp: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _gpu_info():
    if not torch.cuda.is_available():
        return [], [], []
    names, total, free = [], [], []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        names.append(p.name)
        total.append(round(p.total_memory / 1e9, 2))
        try:
            f, _ = torch.cuda.mem_get_info(i)
            free.append(round(f / 1e9, 2))
        except Exception:
            free.append(round(p.total_memory / 1e9, 2))
    return names, total, free


def detect(force_single: bool = False, cpu_only: bool = False) -> Plan:
    names, total, free = ([], [], []) if cpu_only else _gpu_info()
    n = len(names)
    if force_single:
        n = min(n, 1)
        names, total, free = names[:1], total[:1], free[:1]

    cpu_count = os.cpu_count() or 1
    try:
        ram_gb = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except Exception:
        ram_gb = 0.0

    smallest = min(free) if free else 0.0

    # ---- capacity tier from the *smallest* usable GPU (or RAM on CPU) ----
    if n == 0:
        tier, d_model, layers, heads, bs = "tiny", 96, 2, 4, 32
    elif smallest >= 38:
        tier, d_model, layers, heads, bs = "xlarge", 768, 8, 12, 512
    elif smallest >= 20:
        tier, d_model, layers, heads, bs = "large", 512, 6, 8, 256
    elif smallest >= 10:
        tier, d_model, layers, heads, bs = "medium", 320, 4, 8, 128
    elif smallest >= 6:
        tier, d_model, layers, heads, bs = "small", 192, 3, 4, 64
    else:
        tier, d_model, layers, heads, bs = "tiny", 96, 2, 4, 32

    # a single GPU has to hold all three nets at once -> shrink one tier
    if n == 1:
        d_model = max(96, int(d_model * 0.6) // heads * heads)
        layers = max(2, layers - 1)
        bs = max(32, bs // 2)
        tier += "-shared"

    # ---- device placement ----
    if n >= 4:
        dp, da, dt = "cuda:0", "cuda:1", "cuda:2"   # cuda:3 = env/rollout workers
    elif n == 3:
        dp, da, dt = "cuda:0", "cuda:1", "cuda:2"
    elif n == 2:
        dp, da, dt = "cuda:0", "cuda:0", "cuda:1"
    elif n == 1:
        dp = da = dt = "cuda:0"
    else:
        dp = da = dt = "cpu"

    # advisor always on CPU, and never starves the trainers
    adv_threads = max(1, min(4, cpu_count - 2)) if cpu_count > 2 else 1

    return Plan(
        n_gpus=n, gpu_names=names, gpu_mem_gb=total, free_mem_gb=free,
        cpu_count=cpu_count, ram_gb=ram_gb, size_tier=tier,
        dev_predictor=dp, dev_analyzer=da, dev_trader=dt, dev_advisor="cpu",
        d_model=d_model, n_layers=layers, n_heads=heads, batch_size=bs,
        advisor_batch=8 if cpu_count <= 4 else 32, advisor_threads=adv_threads,
        parallel_mode="multiprocess" if n != 1 else "multiprocess",
        amp=(n > 0),
    )


def describe(plan: Plan) -> str:
    L = ["=" * 68, "HARDWARE PLAN", "=" * 68,
         f"host           : {platform.node()}  ({platform.machine()})",
         f"cpu / ram      : {plan.cpu_count} cores, {plan.ram_gb} GB",
         f"gpus           : {plan.n_gpus}"]
    for i, nm in enumerate(plan.gpu_names):
        L.append(f"  [{i}] {nm}  total={plan.gpu_mem_gb[i]} GB  free={plan.free_mem_gb[i]} GB")
    L += [f"size tier      : {plan.size_tier}",
          f"d_model/layers : {plan.d_model} / {plan.n_layers} (heads {plan.n_heads})",
          f"batch size     : {plan.batch_size}   amp={plan.amp}",
          "placement:",
          f"  price_predictor -> {plan.dev_predictor}",
          f"  market_analyzer -> {plan.dev_analyzer}",
          f"  trader (PPO)    -> {plan.dev_trader}",
          f"  llm advisor     -> {plan.dev_advisor} ({plan.advisor_threads} threads)",
          "=" * 68]
    return "\n".join(L)


if __name__ == "__main__":
    p = detect()
    print(describe(p))
