"""
High-dimensional OU weights: stochastic Ornstein-Uhlenbeck drift on S^{d-1}, d >= 4.

Produces only diagnostic plots (G² with MC bands, EDI, H-scaling).
The OU regime satisfies Assumption (62) in the paper, so Theorem 4.16 predicts
O(H^{-1}) decay of the strong upper gradient.
"""

DESCRIPTION = "S^{d-1} OU weights: O(H⁻¹) gradient decay in high dimension"

import os
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from attn_sphere import (
    simulate,
    ou_drift,
    tokens_sd_clustered,
    symmetric_random_weights,
)
from attn_sphere.plot_utils import (
    setup_style,
    plot_energy_balance,
    HEAD_COLORS,
    C_CASE2,
)

_DEFAULTS = dict(T=20.0, dt=5e-3, H=[1, 10, 100], n=200, d=5, n_mc=20, noise_var=2.0)


def run(args):
    setup_style()
    T = getattr(args, "T", None) or _DEFAULTS["T"]
    dt = getattr(args, "dt", None) or _DEFAULTS["dt"]
    heads = getattr(args, "H", None) or _DEFAULTS["H"]
    n_tok = getattr(args, "n", None) or _DEFAULTS["n"]
    d = getattr(args, "d", None) or _DEFAULTS["d"]
    n_mc = getattr(args, "n_mc", None) or _DEFAULTS["n_mc"]
    mc_batch = getattr(args, "mc_batch", None) or min(10, n_mc)
    noise_var = getattr(args, "noise_var", None) or _DEFAULTS["noise_var"]
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

    x0 = tokens_sd_clustered(n=n_tok, d=d, seed=seed)
    rec = 5

    results = {}
    for H in heads:
        D0 = symmetric_random_weights(H, d=d, seed=seed + H, device=device)
        res = simulate(
            x0,
            D0,
            ou_drift,
            noise_var=noise_var,
            device=device,
            T=T,
            dt=dt,
            rec=rec,
            seed=3000 + H,
            n_paths=n_mc,
            mc_batch=mc_batch,
            store_traj=False,
        )
        results[H] = res
        pt_path = os.path.join(results_dir, f"hd_ou_d{d}_H{H}.pt")
        torch.save(
            {
                "hd": res,
                "H": H,
                "d": d,
                "case": "hd_ou",
                "n_mc": n_mc,
                "noise_var": noise_var,
            },
            pt_path,
        )
        print(f"  [hd_ou] d={d}, H={H} done — final G²={res['G2'][-1]:.4e}")

    if not no_plot:
        _plot(results, heads, figures_dir, T, d)


def _plot(results, heads, figures_dir, T, d):
    suffix = f"d{d}"

    # --- G² time series with MC uncertainty bands ---
    fig, ax = plt.subplots(figsize=(7, 4))
    for H, color in zip(heads, HEAD_COLORS):
        res = results[H]
        t = res["t"]
        g2 = np.maximum(res["G2"], 1e-14)
        ax.semilogy(t, g2, color=color, lw=2.0, label=f"$H={H}$")
        if "G2_std" in res:
            lo = np.maximum(g2 - res["G2_std"], 1e-14)
            hi = g2 + res["G2_std"]
            ax.fill_between(t, lo, hi, color=color, alpha=0.15, linewidth=0)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\mathscr{G}_t^2$")
    ax.grid(True, ls=":", alpha=0.3)
    ax.legend()
    ax.set_title(rf"$S^{{{d-1}}}$ OU — metric slope $\mathscr{{G}}^2$ ($\pm 1\sigma$)")
    fig.savefig(
        os.path.join(figures_dir, f"hd_ou_g2_{suffix}.pdf"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)

    # --- EDI balance (largest H) ---
    H_max = max(heads)
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_energy_balance(ax, results[H_max], rf"$S^{{{d-1}}}$ OU EDI ($H={H_max}$)", stoch=True)
    ax.legend(ncol=2, fontsize=8, framealpha=0.9)
    fig.savefig(
        os.path.join(figures_dir, f"hd_ou_edi_{suffix}.pdf"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)

    # --- G² vs H scaling (log-log, should confirm O(H^{-1})) ---
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
    # Reference O(H^{-1}) line
    if len(heads) >= 2:
        c_ref = mean_vals[0] * H_arr[0]
        ax.plot(H_arr, c_ref / H_arr, color=C_CASE2, lw=1.5, ls=":", label=r"$\mathcal{O}(H^{-1})$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$H$")
    ax.set_ylabel(r"$\overline{\mathscr{G}^2}$")
    ax.grid(True, which="both", ls=":", alpha=0.25)
    ax.legend()
    ax.set_title(rf"$S^{{{d-1}}}$ OU — $\overline{{\mathscr{{G}}^2}}$ vs $H$")
    fig.savefig(
        os.path.join(figures_dir, f"hd_ou_scaling_{suffix}.pdf"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
