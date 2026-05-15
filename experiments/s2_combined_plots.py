"""
Combined S² plots: OU vs oscillating (ported from new_plot.py).
"""

DESCRIPTION = "S² combined plots: OU vs oscillating"

import os
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

from attn_sphere.plot_utils import (
    setup_style,
    setup_s2_axis,
    draw_s2_scatter,
    C_CASE1,
    C_CASE2,
    C_SUM,
    C_ENG,
    C_DIS,
    C_ORIG,
)


def _load_case(results_dir, prefix, heads):
    out = {}
    for H in heads:
        path = os.path.join(results_dir, f"{prefix}_H{H}.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing result file: {path}")
        out[H] = torch.load(path, weights_only=False)
    return out


def _get_traj(res):
    return res.get("traj") or res.get("mean_traj")


def _get_snapshot(res, t_target):
    traj = _get_traj(res)
    if traj is None:
        return None
    idx = int(np.argmin(np.abs(res["t"] - t_target)))
    pts = traj[idx]
    if pts.ndim == 3:
        pts = pts[0]
    return pts


def _shade(ax, t, mean, std, color, alpha=0.15):
    ax.fill_between(t, mean - std, mean + std, color=color, alpha=alpha, linewidth=0)


def run(args):
    setup_style()
    results_dir = getattr(args, "results_dir", "./results")
    figures_dir = getattr(args, "figures_dir", "./figures")
    heads = getattr(args, "H", None) or [1, 10, 100]
    if isinstance(heads, int):
        heads = [heads]
    heads = list(heads)

    os.makedirs(figures_dir, exist_ok=True)

    case1 = _load_case(results_dir, "s2_oscillating", heads)
    case2 = _load_case(results_dir, "s2_ou", heads)

    H_max = max(heads)
    res_osc = case1[H_max]["s2"]
    res_ou = case2[H_max]["s2"]

    T = min(res_osc["t"][-1], res_ou["t"][-1])
    snap_times = [0.8 * T, 0.9 * T, T]

    # === Combined snapshots ===
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 4, wspace=0.05, hspace=0.1)
    rows = [
        (res_ou, C_CASE2, "OU (H=max)"),
        (res_osc, C_CASE1, "Oscillating (H=max)"),
    ]

    for row_idx, (res, color, title) in enumerate(rows):
        for col in range(4):
            ax = fig.add_subplot(gs[row_idx, col], projection="3d")
            setup_s2_axis(ax)
            if col == 0:
                pts = _get_snapshot(res, 0.0)
                draw_s2_scatter(ax, pts, color=C_ORIG, size=24)
                ax.set_title("t=0", fontsize=11, pad=2)
            else:
                pts = _get_snapshot(res, snap_times[col - 1])
                draw_s2_scatter(ax, pts, color=color, size=24)
                ax.set_title(f"t={snap_times[col - 1]:.1f}", fontsize=11, pad=2)
            if col == 0:
                ax.text2D(0.02, 0.9, title, transform=ax.transAxes, fontsize=11)

    fig.savefig(os.path.join(figures_dir, "s2_combined_snapshots.pdf"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # === Combined gradient plots ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.subplots_adjust(wspace=0.35, hspace=0.3)

    colors_heads = ["#c0392b", "#e67e22", "#2980b9"]
    head_counts = heads

    # OU time-series
    ax_ou_ts = axes[0, 0]
    mean_vals_ou = []
    for H, color in zip(head_counts, colors_heads):
        res = case2[H]["s2"]
        ax_ou_ts.semilogy(res["t"], np.maximum(res["G2"], 1e-14), color=color, lw=2.0, label=f"H={H}")
        mean_vals_ou.append(np.mean(res["G2"][max(1, len(res["G2"]) // 2):]))
    ax_ou_ts.set_xlabel(r"$t$")
    ax_ou_ts.set_ylabel(r"$\mathscr{G}_t^2$")
    ax_ou_ts.grid(True, alpha=0.25, linestyle=":")
    ax_ou_ts.set_title("OU — time series")

    # OU scaling
    ax_ou_scal = axes[0, 1]
    all_heads = np.array(head_counts, dtype=float)
    all_means = np.array(mean_vals_ou)
    popt_ou, _ = curve_fit(lambda H, a, b: a * H**b, all_heads, all_means)
    H_fit = np.logspace(np.log10(all_heads.min()), np.log10(all_heads.max()), 200)
    ax_ou_scal.scatter(all_heads, all_means, color="#1a5276", s=60, zorder=5)
    ax_ou_scal.plot(H_fit, popt_ou[0] * H_fit**popt_ou[1], color="#7f8c8d", lw=2.0, ls="--")
    ax_ou_scal.set_xscale("log")
    ax_ou_scal.set_yscale("log")
    ax_ou_scal.grid(True, which="both", ls=":", alpha=0.25)
    ax_ou_scal.set_xlabel(r"$H$")
    ax_ou_scal.set_ylabel(r"$\overline{\mathscr{G}^2}$")
    ax_ou_scal.set_title("OU — scaling")

    # Oscillating time-series
    ax_os_ts = axes[1, 0]
    mean_vals_osc = []
    for H, color in zip(head_counts, colors_heads):
        res = case1[H]["s2"]
        ax_os_ts.semilogy(res["t"], np.maximum(res["G2"], 1e-14), color=color, lw=2.0, label=f"H={H}")
        mean_vals_osc.append(np.mean(res["G2"][max(1, len(res["G2"]) // 2):]))
    ax_os_ts.set_xlabel(r"$t$")
    ax_os_ts.set_ylabel(r"$\mathscr{G}_t^2$")
    ax_os_ts.grid(True, alpha=0.25, linestyle=":")
    ax_os_ts.set_title("Oscillating — time series")

    # Oscillating scaling
    ax_os_scal = axes[1, 1]
    all_means_osc = np.array(mean_vals_osc)
    popt_osc, _ = curve_fit(lambda H, a, b: a * H**b, all_heads, all_means_osc)
    ax_os_scal.scatter(all_heads, all_means_osc, color="#1a5276", s=60, zorder=5)
    ax_os_scal.plot(H_fit, popt_osc[0] * H_fit**popt_osc[1], color="#7f8c8d", lw=2.0, ls="--")
    ax_os_scal.set_xscale("log")
    ax_os_scal.set_yscale("log")
    ax_os_scal.grid(True, which="both", ls=":", alpha=0.25)
    ax_os_scal.set_xlabel(r"$H$")
    ax_os_scal.set_ylabel(r"$\overline{\mathscr{G}^2}$")
    ax_os_scal.set_title("Oscillating — scaling")

    handles = [Line2D([0], [0], color=c, lw=2.0, label=f"H={h}") for h, c in zip(head_counts, colors_heads)]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), framealpha=0.9)
    fig.savefig(os.path.join(figures_dir, "s2_combined_gradient.pdf"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # === Combined energy balance ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.subplots_adjust(wspace=0.35)

    for ax, (res, title, stoch) in zip(
        axes,
        [(res_ou, "OU (H=max)", True), (res_osc, "Oscillating (H=max)", False)],
    ):
        t = res["t"]
        delta_E = res["E"] - res["E"][0]
        cum_G2 = res["G2_int"]
        cum_dE = res["dE_int"]
        if stoch:
            cum_add = res["add_int"]
            lhs = delta_E + cum_G2 - cum_dE - cum_add
            if "E_std" in res:
                _shade(ax, t, delta_E, res["E_std"], C_ENG)
            if "G2_int_std" in res:
                _shade(ax, t, cum_G2, res["G2_int_std"], C_DIS)
            if "dE_int_std" in res:
                _shade(ax, t, cum_dE, res["dE_int_std"], C_CASE2)
            if "add_int_std" in res:
                _shade(ax, t, cum_add, res["add_int_std"], "pink")
            ax.plot(t, cum_add, color="pink", lw=2.0, ls="-.", label=r"$M_t$")
        else:
            lhs = delta_E + cum_G2 - cum_dE

        ax.plot(t, delta_E, color=C_ENG, lw=2.2, label=r"$\Delta \mathcal{E}$")
        ax.plot(t, cum_G2, color=C_DIS, lw=2.2, ls="--", label=r"$\int_0^t \mathscr{G}_s^2 ds$")
        ax.plot(t, cum_dE, color=C_CASE2, lw=2.0, ls=":", label=r"$\int_0^t \partial_s \mathcal{E}\,ds$")
        ax.plot(t, lhs, color=C_SUM, lw=3.0, label=r"EDI ($\approx 0$)")
        ax.axhline(0, color="#2c3e50", lw=1.5, alpha=0.6)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, which="both", alpha=0.25, linestyle=":")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, framealpha=0.9)
    fig.savefig(os.path.join(figures_dir, "s2_combined_energy_balance.pdf"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Optional: report correlations
    pearsonr(1.0 / np.array(heads, dtype=float), np.array(mean_vals_ou))
