"""
S² oscillating weights: deterministic rotating drift on the sphere.
"""

DESCRIPTION = "S² oscillating D(t): orbiting clusters on the sphere"

import os
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from attn_sphere import (
    simulate,
    make_drift_oscillating,
    tokens_s2_octahedral,
    symmetric_random_weights,
)
from attn_sphere.plot_utils import (
    setup_style,
    setup_s2_axis,
    draw_s2_scatter,
    plot_energy_balance,
    HEAD_COLORS,
    C_CASE1,
    C_ORIG,
)
from experiments.gif_utils import save_s2_gif

_DEFAULTS = dict(T=20.0, dt=5e-3, H=[1, 10, 100], n=200)


def run(args):
    setup_style()
    T = getattr(args, "T", None) or _DEFAULTS["T"]
    dt = getattr(args, "dt", None) or _DEFAULTS["dt"]
    heads = getattr(args, "H", None) or _DEFAULTS["H"]
    n_tok = getattr(args, "n", None) or _DEFAULTS["n"]
    cpu = getattr(args, "cpu", False)
    no_plot = getattr(args, "no_plot", False)
    make_gif = getattr(args, "gif", False)
    seed = getattr(args, "seed", 42)
    results_dir = getattr(args, "results_dir", "./results")
    figures_dir = getattr(args, "figures_dir", "./figures")

    if isinstance(heads, int):
        heads = [heads]
    heads = list(heads)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    device = (
        torch.device("cpu")
        if (cpu or not torch.cuda.is_available())
        else torch.device("cuda:0")
    )

    drift = make_drift_oscillating(d=3)
    x0 = tokens_s2_octahedral(n=n_tok, seed=seed)
    rec = 5

    results = {}
    for H in heads:
        D0 = symmetric_random_weights(H, d=3, seed=seed + H, device=device)
        res = simulate(
            x0,
            D0,
            drift,
            noise_var=0.0,
            device=device,
            T=T,
            dt=dt,
            rec=rec,
            store_traj=True,
        )
        results[H] = res
        pt_path = os.path.join(results_dir, f"s2_oscillating_H{H}.pt")
        torch.save({"s2": res, "H": H, "case": "s2_oscillating"}, pt_path)

    if not no_plot:
        _plot(results, heads, figures_dir, T)

    if make_gif:
        H = heads[0]
        traj = results[H].get("traj") or results[H].get("mean_traj")
        gif_path = os.path.join(figures_dir, f"s2_oscillating_H{H}.gif")
        save_s2_gif(traj, gif_path, color=C_CASE1, title=f"S2 oscillating (H={H})")


def _plot(results, heads, figures_dir, T):
    snap_times = [0.0, 0.25 * T, 0.5 * T, 0.75 * T, T]
    n_rows = len(heads)
    fig = plt.figure(figsize=(3.2 * len(snap_times), 3.0 * n_rows))
    gs = fig.add_gridspec(n_rows, len(snap_times), wspace=0.05, hspace=0.1)

    for row, H in enumerate(heads):
        res = results[H]
        traj = res.get("traj") or res.get("mean_traj")
        for col, ts in enumerate(snap_times):
            ax = fig.add_subplot(gs[row, col], projection="3d")
            setup_s2_axis(ax)
            idx = int(np.argmin(np.abs(res["t"] - ts)))
            pts = traj[idx] if traj is not None else None
            if pts is not None and pts.ndim == 3:
                pts = pts[0]
            draw_s2_scatter(
                ax,
                pts,
                color=C_ORIG if ts == 0 else HEAD_COLORS[row % len(HEAD_COLORS)],
            )
            if row == 0:
                ax.set_title(f"$t={ts:.1f}$", fontsize=10, pad=2)
            if col == 0:
                ax.text2D(0.02, 0.9, f"H={H}", transform=ax.transAxes, fontsize=10)

    fig.suptitle("S² oscillating weights — snapshots", fontsize=13)
    path = os.path.join(figures_dir, "s2_oscillating_snapshots.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for H, color in zip(heads, HEAD_COLORS):
        ax.semilogy(
            results[H]["t"],
            np.maximum(results[H]["G2"], 1e-14),
            color=color,
            lw=2.0,
            label=f"H={H}",
        )
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\mathscr{G}_t^2$")
    ax.grid(True, ls=":", alpha=0.3)
    ax.legend()
    ax.set_title("S² oscillating — metric slope")
    path2 = os.path.join(figures_dir, "s2_oscillating_g2.pdf")
    fig.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    plot_energy_balance(ax, results[heads[0]], "S² oscillating EDI", stoch=False)
    ax.legend(ncol=2, fontsize=8, framealpha=0.9)
    path3 = os.path.join(figures_dir, "s2_oscillating_edi.pdf")
    fig.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close(fig)
