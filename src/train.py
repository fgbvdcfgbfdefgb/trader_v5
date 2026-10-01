#!/usr/bin/env python3
"""
trader_v5 - fully offline multi-agent RL training.

Three agents train CONCURRENTLY in separate processes and feed each other:

  [producer]  samples a random trading day, builds features, and runs the
              CPU LLM news advisor point-in-time over that day
        |
        +--> [price_predictor]  (GPU A)  supervised multi-horizon returns
        +--> [market_analyzer]  (GPU B)  regime + vol, fused with news
        +--> [trader / PPO]     (GPU C)  consumes BOTH agents' live latents

The predictor and analyzer publish fresh weights every `--sync-every` epochs;
the trader hot-reloads them, so the policy is always trading on top of the
current beliefs of the other two.

Everything reads from the repo. No network access is required at run time.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import queue
import sys
import time
import traceback

import numpy as np
import torch
import torch.multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import resources                                     # noqa: E402
from dataset import MarketData, forward_targets, ASSETS, N_TIME_FEAT  # noqa: E402
from advisor import LLMAdvisor, N_ADVISOR_FEAT       # noqa: E402
from env import TradingEnv, regime_labels, realised_vol, START_CASH, TARGET  # noqa: E402
from nets import (PricePredictor, MarketAnalyzer, TraderPolicy, HORIZONS,
                  count_params)                      # noqa: E402
import viz                                           # noqa: E402

PROJ_DIM = 32


def ser(sd):
    b = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in sd.items()}, b)
    return b.getvalue()


def deser(buf):
    return torch.load(io.BytesIO(buf), map_location="cpu", weights_only=True)


def projection(d, out=PROJ_DIM, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(d, out, generator=g) / (d ** 0.5)


# ----------------------------------------------------------------------------
# producer
# ----------------------------------------------------------------------------
def producer_proc(cfg, qs, stop, nepochs):
    try:
        rng = np.random.default_rng(cfg["seed"])
        md = MarketData(os.path.join(ROOT, "data", "market"), ASSETS)
        days = md.days
        print(f"[producer] {len(days)} tradable days "
              f"({days[0]} .. {days[-1]})", flush=True)
        adv = LLMAdvisor(
            os.path.join(ROOT, "models"), os.path.join(ROOT, "data", "news"),
            os.path.join(ROOT, "outputs", "advisor_cache"),
            threads=cfg["advisor_threads"], batch=cfg["advisor_batch"],
            enabled=cfg["use_advisor"])
        for ep in range(nepochs):
            if stop.is_set():
                break
            for _ in range(12):
                day = md.sample_day(rng, cfg["day_lo"], cfg["day_hi"])
                e = md.episode(day)
                if e is not None:
                    break
            if e is None:
                continue
            t0 = time.time()
            av, notes = adv.day_track(day, e["timestamps"], ASSETS,
                                      every=cfg["advisor_every"])
            pack = {
                "epoch": ep, "day": day,
                "features": e["features"], "prices": e["prices"],
                "warmup": e["warmup"], "advisor": av, "notes": notes,
                "adv_secs": round(time.time() - t0, 2),
            }
            for q in qs:
                try:
                    q.put(pack, timeout=600)
                except Exception:
                    pass
        for q in qs:
            try:
                q.put(None, timeout=60)
            except Exception:
                pass
        print("[producer] done", flush=True)
    except Exception:
        traceback.print_exc()
        stop.set()


# ----------------------------------------------------------------------------
# price predictor
# ----------------------------------------------------------------------------
def predictor_proc(cfg, q, shared, stop):
    try:
        dev = torch.device(cfg["dev_predictor"])
        net = PricePredictor(cfg["n_feat"], len(ASSETS),
                             cfg["d_model"], cfg["n_layers"]).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"] and dev.type == "cuda")
        print(f"[predictor] {count_params(net)/1e6:.2f}M params on {dev}", flush=True)
        shared["pred_cfg"] = (cfg["n_feat"], len(ASSETS), cfg["d_model"], cfg["n_layers"])
        n = 0
        while not stop.is_set():
            try:
                pk = q.get(timeout=300)
            except queue.Empty:
                continue
            if pk is None:
                break
            x = torch.from_numpy(pk["features"]).unsqueeze(0).to(dev)
            y = torch.from_numpy(
                forward_targets(pk["prices"], HORIZONS)).unsqueeze(0).to(dev)
            net.train()
            tot = 0.0
            for _ in range(cfg["inner_steps"]):
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=cfg["amp"] and dev.type == "cuda"):
                    loss = net.loss(x, y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(opt); scaler.update()
                tot += float(loss.detach())
            n += 1
            shared["pred_loss"] = tot / cfg["inner_steps"]
            if n % cfg["sync_every"] == 0 or n == 1:
                shared["pred_w"] = ser(net.state_dict())
                shared["pred_v"] = n
            if n % 10 == 0:
                print(f"[predictor] ep{pk['epoch']} nll={tot/cfg['inner_steps']:.4f}",
                      flush=True)
        shared["pred_w"] = ser(net.state_dict()); shared["pred_v"] = n + 1
        torch.save(net.state_dict(), os.path.join(cfg["ckpt"], "price_predictor.pt"))
        print("[predictor] done", flush=True)
    except Exception:
        traceback.print_exc(); stop.set()


# ----------------------------------------------------------------------------
# market analyzer
# ----------------------------------------------------------------------------
def analyzer_proc(cfg, q, shared, stop):
    try:
        dev = torch.device(cfg["dev_analyzer"])
        net = MarketAnalyzer(cfg["n_feat"], N_ADVISOR_FEAT, len(ASSETS),
                             cfg["d_model"], max(2, cfg["n_layers"] - 1)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"] and dev.type == "cuda")
        print(f"[analyzer ] {count_params(net)/1e6:.2f}M params on {dev}", flush=True)
        shared["anal_cfg"] = (cfg["n_feat"], N_ADVISOR_FEAT, len(ASSETS),
                              cfg["d_model"], max(2, cfg["n_layers"] - 1))
        n = 0
        while not stop.is_set():
            try:
                pk = q.get(timeout=300)
            except queue.Empty:
                continue
            if pk is None:
                break
            x = torch.from_numpy(pk["features"]).unsqueeze(0).to(dev)
            a = torch.from_numpy(pk["advisor"]).unsqueeze(0).to(dev)
            ry = torch.from_numpy(regime_labels(pk["prices"])).unsqueeze(0).to(dev)
            vy = torch.from_numpy(realised_vol(pk["prices"])).unsqueeze(0).to(dev)
            net.train()
            tot = 0.0
            for _ in range(cfg["inner_steps"]):
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=cfg["amp"] and dev.type == "cuda"):
                    loss, ce, mse = net.loss(x, a, ry, vy)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(opt); scaler.update()
                tot += float(loss.detach())
            n += 1
            shared["anal_loss"] = tot / cfg["inner_steps"]
            if n % cfg["sync_every"] == 0 or n == 1:
                shared["anal_w"] = ser(net.state_dict())
                shared["anal_v"] = n
            if n % 10 == 0:
                print(f"[analyzer ] ep{pk['epoch']} loss={tot/cfg['inner_steps']:.4f}",
                      flush=True)
        shared["anal_w"] = ser(net.state_dict()); shared["anal_v"] = n + 1
        torch.save(net.state_dict(), os.path.join(cfg["ckpt"], "market_analyzer.pt"))
        print("[analyzer ] done", flush=True)
    except Exception:
        traceback.print_exc(); stop.set()


# ----------------------------------------------------------------------------
# trader (PPO)
# ----------------------------------------------------------------------------
def trader_proc(cfg, q, shared, stop):
    try:
        dev = torch.device(cfg["dev_trader"])
        nA = len(ASSETS)
        d = cfg["d_model"]
        pp = PricePredictor(cfg["n_feat"], nA, d, cfg["n_layers"]).to(dev).eval()
        ma = MarketAnalyzer(cfg["n_feat"], N_ADVISOR_FEAT, nA, d,
                            max(2, cfg["n_layers"] - 1)).to(dev).eval()
        Pp = projection(d, PROJ_DIM, 1).to(dev)
        Pm = projection(d, PROJ_DIM, 2).to(dev)

        n_pred = nA * len(HORIZONS) * 2 + PROJ_DIM
        n_reg = nA * 3 + nA + PROJ_DIM
        n_port = nA + 4
        obs_dim = cfg["n_feat"] + n_pred + n_reg + n_port + N_ADVISOR_FEAT

        pol = TraderPolicy(cfg["n_feat"] + N_ADVISOR_FEAT, n_pred, n_reg,
                           n_port, nA, d, 2).to(dev)
        opt = torch.optim.AdamW(pol.parameters(), lr=cfg["lr"], eps=1e-5)
        print(f"[trader   ] {count_params(pol)/1e6:.2f}M params on {dev} "
              f"obs_dim={obs_dim}", flush=True)

        pv = av = -1
        equities, returns, hits = [], [], []
        losses = {"predictor": [], "analyzer": [], "policy": [], "value": []}
        outdir = cfg["outdir"]; os.makedirs(outdir, exist_ok=True)
        metrics_path = os.path.join(outdir, "metrics.jsonl")
        n = 0

        while not stop.is_set():
            try:
                pk = q.get(timeout=300)
            except queue.Empty:
                continue
            if pk is None:
                break

            # ---- hot-reload partner weights ----
            if shared.get("pred_v", -1) > pv:
                try:
                    pp.load_state_dict(deser(shared["pred_w"])); pv = shared["pred_v"]
                except Exception:
                    pass
            if shared.get("anal_v", -1) > av:
                try:
                    ma.load_state_dict(deser(shared["anal_w"])); av = shared["anal_v"]
                except Exception:
                    pass

            X = torch.from_numpy(pk["features"]).unsqueeze(0).to(dev)
            A = torch.from_numpy(pk["advisor"]).unsqueeze(0).to(dev)

            # ---- one causal pass from each partner over the whole day ----
            with torch.no_grad():
                mu, ls, hp = pp(X)
                rl, vol, hm = ma(X, A)
                T = X.shape[1]
                pred_feat = torch.cat([
                    mu.reshape(T, -1), ls.reshape(T, -1), hp[0] @ Pp], -1)
                reg_prob = torch.softmax(rl[0], -1).reshape(T, -1)
                reg_feat = torch.cat([reg_prob, vol[0], hm[0] @ Pm], -1)
            pred_np = pred_feat.float().cpu().numpy()
            reg_np = reg_feat.float().cpu().numpy()
            pred_track = mu[0, :, 2, :].float().cpu().numpy()      # 15m horizon
            reg_track = torch.softmax(rl[0, :, 0, :], -1).float().cpu().numpy()

            env = TradingEnv(
                {"prices": pk["prices"], "features": pk["features"],
                 "warmup": pk["warmup"]},
                pk["advisor"], decision_every=cfg["decision_every"],
                fee=cfg["fee"], max_gross=cfg["max_gross"])
            o = env.reset()

            # ---- rollout ----
            O, ACT, LOGP, VAL, REW, DONE = [], [], [], [], [], []
            hx = None
            while True:
                t = o["t"]
                obs = np.concatenate([o["market"], o["advisor"],
                                      pred_np[t], reg_np[t], o["portfolio"]])
                ot = torch.from_numpy(obs).float().unsqueeze(0).to(dev)
                with torch.no_grad():
                    a, lp, v, hx = pol.act(ot, hx)
                o2, r, done, info = env.step(a[0].cpu().numpy())
                O.append(obs); ACT.append(a[0].cpu().numpy())
                LOGP.append(float(lp)); VAL.append(float(v))
                REW.append(r); DONE.append(done)
                o = o2
                if done:
                    break

            # ---- GAE ----
            R = np.asarray(REW, np.float32); V = np.asarray(VAL, np.float32)
            adv_buf = np.zeros_like(R); last = 0.0
            for i in reversed(range(len(R))):
                nv = 0.0 if i == len(R) - 1 else V[i + 1]
                delta = R[i] + cfg["gamma"] * nv - V[i]
                last = delta + cfg["gamma"] * cfg["lam"] * last
                adv_buf[i] = last
            ret = adv_buf + V
            adv_t = torch.from_numpy(
                (adv_buf - adv_buf.mean()) / (adv_buf.std() + 1e-8)).to(dev)
            ret_t = torch.from_numpy(ret).to(dev)
            obs_t = torch.from_numpy(np.asarray(O, np.float32)).to(dev)
            act_t = torch.from_numpy(np.asarray(ACT, np.float32)).to(dev)
            old_lp = torch.from_numpy(np.asarray(LOGP, np.float32)).to(dev)

            # ---- PPO update (BPTT over the day) ----
            pl_acc = vl_acc = 0.0
            for _ in range(cfg["ppo_epochs"]):
                hx2 = None; lps = []; ents = []; vs = []
                for i in range(obs_t.shape[0]):
                    lp_i, ent_i, v_i, hx2 = pol.evaluate(
                        obs_t[i:i + 1], act_t[i:i + 1], hx2)
                    lps.append(lp_i); ents.append(ent_i); vs.append(v_i)
                lp_all = torch.cat(lps); ent = torch.cat(ents).mean()
                v_all = torch.cat(vs)
                ratio = (lp_all - old_lp).clamp(-10, 10).exp()
                s1 = ratio * adv_t
                s2 = ratio.clamp(1 - cfg["clip"], 1 + cfg["clip"]) * adv_t
                p_loss = -torch.min(s1, s2).mean()
                v_loss = torch.nn.functional.smooth_l1_loss(v_all, ret_t)
                loss = p_loss + cfg["vf_coef"] * v_loss - cfg["ent_coef"] * ent
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pol.parameters(), 0.5)
                opt.step()
                pl_acc += float(p_loss.detach()); vl_acc += float(v_loss.detach())

            n += 1
            equities.append(info["equity"]); returns.append(info["return_pct"])
            hits.append(1.0 if info["hit_target"] else 0.0)
            losses["policy"].append(pl_acc / cfg["ppo_epochs"])
            losses["value"].append(vl_acc / cfg["ppo_epochs"])
            losses["predictor"].append(float(shared.get("pred_loss", 0.0)))
            losses["analyzer"].append(float(shared.get("anal_loss", 0.0)))

            rec = {"epoch": pk["epoch"], "day": pk["day"], **info,
                   "pred_loss": losses["predictor"][-1],
                   "anal_loss": losses["analyzer"][-1],
                   "policy_loss": losses["policy"][-1],
                   "value_loss": losses["value"][-1],
                   "advisor_secs": pk["adv_secs"],
                   "hit_rate_100": float(np.mean(hits[-100:]))}
            with open(metrics_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[trader   ] ep{pk['epoch']:<5} {pk['day']}  "
                  f"${info['equity']:7.2f} ({info['return_pct']:+7.2f}%)  "
                  f"{'TARGET' if info['hit_target'] else '      '}  "
                  f"sharpe={info['sharpe']:+.2f}  hit100={rec['hit_rate_100']:.2%}",
                  flush=True)

            if pk["epoch"] % cfg["png_every"] == 0:
                try:
                    viz.save_epoch_png(
                        os.path.join(outdir, "epochs",
                                     f"epoch_{pk['epoch']:06d}_{pk['day']}.png"),
                        pk["epoch"], pk["day"], env.hist, info, ASSETS,
                        pk["notes"], losses, equities,
                        regime_track=reg_track[env.hist["t"]],
                        pred_track=pred_track[env.hist["t"]],
                        sentiment_track=pk["advisor"][env.hist["t"]])
                except Exception:
                    traceback.print_exc()
            if n % cfg["ckpt_every"] == 0:
                torch.save(pol.state_dict(),
                           os.path.join(cfg["ckpt"], "trader_policy.pt"))

        torch.save(pol.state_dict(), os.path.join(cfg["ckpt"], "trader_policy.pt"))
        if equities:
            viz.save_summary_png(os.path.join(outdir, "summary.png"),
                                 equities, returns, hits)
        print("[trader   ] done", flush=True)
    except Exception:
        traceback.print_exc(); stop.set()


# ----------------------------------------------------------------------------
def build_cfg(args, plan):
    n_feat = len(ASSETS) * 12 + N_TIME_FEAT
    return {
        "seed": args.seed, "n_feat": n_feat,
        "d_model": args.d_model or plan.d_model,
        "n_layers": args.n_layers or plan.n_layers,
        "amp": plan.amp and not args.no_amp,
        "dev_predictor": args.device or plan.dev_predictor,
        "dev_analyzer": args.device or plan.dev_analyzer,
        "dev_trader": args.device or plan.dev_trader,
        "advisor_threads": plan.advisor_threads,
        "advisor_batch": plan.advisor_batch,
        "advisor_every": args.advisor_every,
        "use_advisor": not args.no_advisor,
        "inner_steps": args.inner_steps, "sync_every": args.sync_every,
        "decision_every": args.decision_every, "fee": args.fee,
        "max_gross": args.max_gross, "gamma": 0.995, "lam": 0.95,
        "clip": 0.2, "vf_coef": 0.5, "ent_coef": args.ent_coef,
        "ppo_epochs": args.ppo_epochs, "lr": args.lr,
        "png_every": args.png_every, "ckpt_every": args.ckpt_every,
        "outdir": args.outdir, "ckpt": args.ckpt,
        "day_lo": args.day_lo, "day_hi": args.day_hi,
    }


def main():
    p = argparse.ArgumentParser("trader_v5 offline trainer")
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d-model", type=int, default=0)
    p.add_argument("--n-layers", type=int, default=0)
    p.add_argument("--device", default="")
    p.add_argument("--single-gpu", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-advisor", action="store_true")
    p.add_argument("--advisor-every", type=int, default=60)
    p.add_argument("--inner-steps", type=int, default=2)
    p.add_argument("--sync-every", type=int, default=5)
    p.add_argument("--decision-every", type=int, default=5)
    p.add_argument("--fee", type=float, default=0.0004)
    p.add_argument("--max-gross", type=float, default=1.0)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--png-every", type=int, default=1)
    p.add_argument("--ckpt-every", type=int, default=25)
    p.add_argument("--outdir", default=os.path.join(ROOT, "outputs"))
    p.add_argument("--ckpt", default=os.path.join(ROOT, "outputs", "checkpoints"))
    p.add_argument("--day-lo", default="")
    p.add_argument("--day-hi", default="")
    p.add_argument("--queue-size", type=int, default=4)
    args = p.parse_args()

    plan = resources.detect(force_single=args.single_gpu, cpu_only=args.cpu)
    print(resources.describe(plan), flush=True)
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.ckpt, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "epochs"), exist_ok=True)
    cfg = build_cfg(args, plan)
    with open(os.path.join(args.outdir, "run_config.json"), "w") as f:
        json.dump({"plan": plan.__dict__, "cfg": cfg,
                   "argv": " ".join(sys.argv)}, f, indent=2, default=str)
    print(json.dumps({k: cfg[k] for k in
                      ("d_model", "n_layers", "amp", "dev_predictor",
                       "dev_analyzer", "dev_trader")}, indent=2), flush=True)

    ctx = mp.get_context("spawn")
    mgr = ctx.Manager()
    shared = mgr.dict()
    stop = ctx.Event()
    qp, qa, qt = (ctx.Queue(args.queue_size) for _ in range(3))

    procs = [
        ctx.Process(target=producer_proc, args=(cfg, [qp, qa, qt], stop, args.epochs), name="producer"),
        ctx.Process(target=predictor_proc, args=(cfg, qp, shared, stop), name="predictor"),
        ctx.Process(target=analyzer_proc, args=(cfg, qa, shared, stop), name="analyzer"),
        ctx.Process(target=trader_proc, args=(cfg, qt, shared, stop), name="trader"),
    ]
    t0 = time.time()
    for pr in procs:
        pr.start()
    try:
        for pr in procs:
            pr.join()
    except KeyboardInterrupt:
        stop.set()
        for pr in procs:
            pr.terminate()
    print(f"\nTRAINING COMPLETE in {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"artifacts -> {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
