"""
Per-epoch diagnostic PNG: everything about one trading day on a single sheet,
including what the news advisor said about that day.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"figure.dpi": 110, "font.size": 8,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.facecolor": "#11151c", "figure.facecolor": "#0b0e13",
                     "text.color": "#dde3ea", "axes.labelcolor": "#dde3ea",
                     "xtick.color": "#9aa6b2", "ytick.color": "#9aa6b2",
                     "axes.edgecolor": "#2a3340", "grid.color": "#2a3340"})

CLR = {"BTCUSDT": "#f7931a", "ETHUSDT": "#8a92ff", "LTCUSDT": "#4fd1c5"}


def save_epoch_png(path, epoch, day, env_hist, info, assets, advisor_notes,
                   losses, equity_curve_all=None, start_cash=20.0, target=30.0,
                   regime_track=None, pred_track=None, sentiment_track=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    eq = np.asarray(env_hist["equity"], float)
    W = np.asarray(env_hist["weights"], float)
    P = np.asarray(env_hist["price"], float)
    R = np.asarray(env_hist["reward"], float)
    DD = np.asarray(env_hist["drawdown"], float) * 100
    TO = np.asarray(env_hist["turnover"], float)
    x = np.arange(len(eq))

    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(6, 3, hspace=0.62, wspace=0.24,
                          height_ratios=[1.25, 1.0, 1.0, 1.0, 1.0, 0.85])

    hit = "HIT" if info.get("hit_target") else "miss"
    fig.suptitle(
        f"trader_v5  |  epoch {epoch}  |  day {day}  |  "
        f"${start_cash:.2f} -> ${info['equity']:.2f}  "
        f"({info['return_pct']:+.2f}%)  target ${target:.0f} {hit}  "
        f"Sharpe {info['sharpe']:.2f}  maxDD {info['max_dd']*100:.2f}%",
        fontsize=13, y=0.975, color="#ffffff")

    # 1 equity
    ax = fig.add_subplot(gs[0, :])
    ax.plot(x, eq, color="#39d98a", lw=1.7, label="equity")
    ax.axhline(start_cash, color="#7a8695", ls="--", lw=0.9, label="start $20")
    ax.axhline(target, color="#ffd166", ls="--", lw=0.9, label="target $30")
    ax.fill_between(x, start_cash, eq, where=eq >= start_cash,
                    color="#39d98a", alpha=0.13)
    ax.fill_between(x, start_cash, eq, where=eq < start_cash,
                    color="#ff6b6b", alpha=0.13)
    ax.set_title("portfolio equity"); ax.set_ylabel("USD")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.2)

    # 2 normalised prices
    ax = fig.add_subplot(gs[1, 0])
    for i, a in enumerate(assets):
        ax.plot(x, P[:, i] / P[0, i] * 100, lw=1.2, color=CLR.get(a), label=a[:3])
    ax.set_title("asset prices (rebased=100)"); ax.legend(fontsize=7, framealpha=0.2)

    # 3 weights
    ax = fig.add_subplot(gs[1, 1])
    for i, a in enumerate(assets):
        ax.plot(x, W[:, i], lw=1.1, color=CLR.get(a), label=a[:3])
    ax.axhline(0, color="#7a8695", lw=0.8)
    ax.set_ylim(-1.05, 1.05); ax.set_title("target weights (neg = short)")
    ax.legend(fontsize=7, framealpha=0.2)

    # 4 gross exposure + turnover
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(x, np.abs(W).sum(1), lw=1.1, color="#8a92ff", label="gross")
    ax.plot(x, np.cumsum(TO) / max(len(TO), 1), lw=1.0, color="#ffd166",
            label="avg turnover")
    ax.set_title("exposure & turnover"); ax.legend(fontsize=7, framealpha=0.2)

    # 5 reward
    ax = fig.add_subplot(gs[2, 0])
    ax.bar(x, R, color=np.where(R >= 0, "#39d98a", "#ff6b6b"), width=1.0)
    ax.set_title("step reward")

    # 6 cumulative reward
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(x, np.cumsum(R), color="#4fd1c5", lw=1.3)
    ax.set_title("cumulative reward")

    # 7 drawdown
    ax = fig.add_subplot(gs[2, 2])
    ax.fill_between(x, DD, 0, color="#ff6b6b", alpha=0.45)
    ax.set_title("drawdown %")

    # 8 predictor
    ax = fig.add_subplot(gs[3, 0])
    if pred_track is not None and len(pred_track):
        pt = np.asarray(pred_track)
        for i, a in enumerate(assets):
            ax.plot(pt[:, i], lw=1.0, color=CLR.get(a), label=a[:3])
        ax.axhline(0, color="#7a8695", lw=0.8)
        ax.legend(fontsize=7, framealpha=0.2)
    ax.set_title("price predictor: 15m forward return (%)")

    # 9 regime
    ax = fig.add_subplot(gs[3, 1])
    if regime_track is not None and len(regime_track):
        rt = np.asarray(regime_track)
        ax.stackplot(np.arange(len(rt)), rt[:, 0], rt[:, 1], rt[:, 2],
                     colors=["#ff6b6b", "#7a8695", "#39d98a"],
                     labels=["bear", "chop", "bull"], alpha=0.85)
        ax.legend(fontsize=7, loc="upper left", framealpha=0.2)
        ax.set_ylim(0, 1)
    ax.set_title("market analyzer: regime belief")

    # 10 sentiment
    ax = fig.add_subplot(gs[3, 2])
    if sentiment_track is not None and len(sentiment_track):
        st = np.asarray(sentiment_track)
        for i, a in enumerate(assets):
            if st.shape[1] > i * 6:
                ax.plot(st[:, i * 6], lw=1.1, color=CLR.get(a), label=a[:3])
        ax.axhline(0, color="#7a8695", lw=0.8)
        ax.set_ylim(-1.05, 1.05); ax.legend(fontsize=7, framealpha=0.2)
    ax.set_title("LLM advisor: news sentiment")

    # 11 losses
    ax = fig.add_subplot(gs[4, 0])
    for k, c in (("predictor", "#f7931a"), ("analyzer", "#8a92ff"),
                 ("policy", "#39d98a"), ("value", "#ff6b6b")):
        v = losses.get(k, [])
        if len(v):
            v = np.asarray(v, float)
            v = (v - v.min()) / (v.ptp() + 1e-9)
            ax.plot(v, lw=1.1, color=c, label=k)
    ax.set_title("training losses (min-max normalised)")
    ax.legend(fontsize=7, framealpha=0.2)

    # 12 equity across epochs
    ax = fig.add_subplot(gs[4, 1])
    if equity_curve_all:
        e = np.asarray(equity_curve_all, float)
        ax.plot(e, lw=1.0, color="#4fd1c5", alpha=0.8)
        if len(e) >= 10:
            ax.plot(np.convolve(e, np.ones(10) / 10, "valid"),
                    lw=1.8, color="#ffd166", label="MA10")
            ax.legend(fontsize=7, framealpha=0.2)
        ax.axhline(target, color="#ffd166", ls="--", lw=0.9)
        ax.axhline(start_cash, color="#7a8695", ls="--", lw=0.9)
    ax.set_title("final equity per epoch")

    # 13 per-asset contribution
    ax = fig.add_subplot(gs[4, 2])
    rets = np.diff(P, axis=0) / P[:-1]
    contrib = (W[:-1] * rets).sum(0) * 100
    ax.bar([a[:3] for a in assets], contrib,
           color=[CLR.get(a) for a in assets])
    ax.axhline(0, color="#7a8695", lw=0.8)
    ax.set_title("per-asset return contribution (%)")

    # 14 advisor text
    ax = fig.add_subplot(gs[5, :]); ax.axis("off")
    lines = ["LLM ADVISOR - point-in-time news visible to the agent on "
             f"{day} (nothing published after each timestamp):"]
    for ts, note in (advisor_notes or [])[:7]:
        lines.append(f"  {str(ts)[:16]}  {note[:155]}")
    if len(advisor_notes or []) == 0:
        lines.append("  (no news available in window)")
    ax.text(0.005, 0.97, "\n".join(lines), va="top", ha="left",
            family="monospace", fontsize=7.0, color="#b9c4d0",
            transform=ax.transAxes)

    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def save_summary_png(path, equities, returns, hits, start_cash=20.0, target=30.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("trader_v5 - training summary", fontsize=13, color="#fff")
    e = np.asarray(equities, float)
    a = axes[0, 0]
    a.plot(e, lw=0.9, color="#4fd1c5", alpha=0.7)
    if len(e) >= 20:
        a.plot(np.convolve(e, np.ones(20) / 20, "valid"), lw=2, color="#ffd166")
    a.axhline(target, color="#ffd166", ls="--"); a.axhline(start_cash, color="#7a8695", ls="--")
    a.set_title("final equity per epoch")
    a = axes[0, 1]
    a.hist(returns, bins=40, color="#8a92ff", alpha=0.85)
    a.axvline(0, color="#7a8695"); a.axvline(50, color="#ffd166", ls="--")
    a.set_title("daily return distribution (%)")
    a = axes[1, 0]
    h = np.asarray(hits, float)
    if len(h) >= 20:
        a.plot(np.convolve(h, np.ones(20) / 20, "valid") * 100, lw=1.6, color="#39d98a")
    a.set_title("rolling-20 hit rate on $30 target (%)"); a.set_ylim(0, 100)
    a = axes[1, 1]
    a.plot(np.cumsum(np.log(np.clip(e / start_cash, 1e-6, None))), lw=1.4, color="#f7931a")
    a.set_title("cumulative log growth")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path
