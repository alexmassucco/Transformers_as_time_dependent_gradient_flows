"""
High-dimensional oscillating weights: deterministic rotating drift on S^{d-1}, d >= 4.

Produces only diagnostic plots (G², EDI, H-scaling); no point-cloud visualisation.
"""

DESCRIPTION = "S^{d-1} oscillating D(t): metric slope and EDI diagnostics in high dimension"

import os
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from attn_sphere import (
    simulate,
    make_drift_oscillating,
    tokens_sd_clustered,
    symmetric_random_weights,
)
from attn_sphere.plot_utils import (
    setup_style,
    plot_energy_balance,
    HEAD_COLORS,
    C_CASE1,
)

_DEFAULTS = dict(T=20.0, dt=5e-3, H=[1, 10, 100], n=200, d=5)


def run(args):
    setup_style()
    T = getattr(args, "T", None) or _DEFAULTS["T"]
    dt = getattr(args, "dt", None) or _DEFAULTS["dt"]
    heads = getattr(args, "H", None) or _DEFAULTS["H"]
    n_tok = getattr(args, "n", None) or _DEFAULTS["n"]
    d = getattr(args, "d", None) or _DEFAULTS["d"]
    cpu = getattr(args, "cpu", False)
    no_plot = getattr(args, "no_plot", False)
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

    drift = make_drift_oscillating(d=d)
    x0 = tokens_sd_clustered(n=n_tok, d=d, seed=seed)
    rec = 5

    results = {}
    for H in heads:
        D0 = symmetric_random_weights(H, d=d, seed=seed + H, device=device)
        res = simulate(
            x0,
            D0,
            drift,
            noise_var=0.0,
            device=device,
            T=T,
            dt=dt,
            rec=rec,
            store_traj=False,
        )
        results[H] = res
        pt_path = os.path.join(results_dir, f"hd_oscillating_d{d}_H{H}.pt")
        torch.save({"hd": res, "H": H, "d": d, "case": "hd_oscillating"}, pt_path)
        print(f"  [hd_oscillating] d={d}, H={H} done — final G²={res['G2'][-1]:.4e}")

    if not no_plot:
        _plot(results, heads, figures_dir, T, d)


def _plot(results, heads, figures_dir, T, d):
    suffix = f"d{d}"

    # --- G² time series ---
    fig, ax = plt.subplots(figsize=(7, 4))
    for H, color in zip(heads, HEAD_COLORS):
        ax.semilogy(
            results[H]["t"],
            np.maximum(results[H]["G2"], 1e-14),
            color=color,
            lw=2.0,
            label=f"$H={H}$",
        )
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\mathscr{G}_t^2$")
    ax.grid(True, ls=":", alpha=0.3)
    ax.legend()
    ax.set_title(rf"$S^{{{d-1}}}$ oscillating — metric slope $\mathscr{{G}}^2$")
    fig.savefig(
        os.path.join(figures_dir, f"hd_oscillating_g2_{suffix}.pdf"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)

    # --- EDI balance (largest H) ---
    H_max = max(heads)
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_energy_balance(ax, results[H_max], rf"$S^{{{d-1}}}$ oscillating EDI ($H={H_max}$)", stoch=False)
    ax.legend(ncol=2, fontsize=8, framealpha=0.9)
    fig.savefig(
        os.path.join(figures_dir, f"hd_oscillating_edi_{suffix}.pdf"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)

    # --- G² vs H scaling ---
    half = max(1, len(results[heads[0]]["G2"]) // 2)
    mean_vals = np.array([np.mean(results[H]["G2"][half:]) for H in heads])
    H_arr = np.array(heads, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(H_arr, mean_vals, color="#1a5276", s=60, zorder=5, label=r"$\overline{\mathscr{G}^2}$")
    if len(heads) >= 2:
        try:
            popt, _ = curve_fit(lambda H, a, b: a * H ** b, H_arr, mean_vals)
            H_fit = np.logspace(np.log10(H_arr.min()), np.log10(H_arr.max()), 200)
            ax.plot(
                H_fit, popt[0] * H_fit ** popt[1],
                color="#7f8c8d", lw=2.0, ls="--",
                label=rf"$a H^b$, $b={popt[1]:.3f}$",
            )
        except RuntimeError:
            pass
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$H$")
    ax.set_ylabel(r"$\overline{\mathscr{G}^2}$")
    ax.grid(True, which="both", ls=":", alpha=0.25)
    ax.legend()
    ax.set_title(rf"$S^{{{d-1}}}$ oscillating — $\overline{{\mathscr{{G}}^2}}$ scaling")
    fig.savefig(
        os.path.join(figures_dir, f"hd_oscillating_scaling_{suffix}.pdf"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
