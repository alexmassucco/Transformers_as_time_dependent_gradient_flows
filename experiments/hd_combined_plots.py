"""
Combined high-dimensional plots: OU vs oscillating on S^{d-1}.

Requires hd_oscillating and hd_ou results to exist first:
    python main.py run hd_oscillating hd_ou --cpu
    python main.py run hd_combined_plots --cpu
"""

DESCRIPTION = "S^{d-1} combined plots: OU vs oscillating diagnostics"

import os
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

from attn_sphere.plot_utils import (
    setup_style,
    C_CASE1,
    C_CASE2,
    C_SUM,
    C_ENG,
    C_DIS,
)

HEAD_COLORS_LOCAL = ["#c0392b", "#e67e22", "#2980b9"]


def _load_case(results_dir, prefix, heads, d):
    out = {}
    for H in heads:
        path = os.path.join(results_dir, f"{prefix}_d{d}_H{H}.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing result file: {path}\n"
                f"Run `python main.py run {prefix.replace('_d', '')} --cpu` first."
            )
        data = torch.load(path, weights_only=False)
        out[H] = data["hd"]
    return out


def _shade(ax, t, mean, std, color, alpha=0.15):
    ax.fill_between(t, mean - std, mean + std, color=color, alpha=alpha, linewidth=0)


def run(args):
    setup_style()
    results_dir = getattr(args, "results_dir", "./results")
    figures_dir = getattr(args, "figures_dir", "./figures")
    heads = getattr(args, "H", None) or [1, 10, 100]
    d = getattr(args, "d", None) or 5
    if isinstance(heads, int):
        heads = [heads]
    heads = list(heads)

    os.makedirs(figures_dir, exist_ok=True)

    ou_results = _load_case(results_dir, "hd_ou", heads, d)
    osc_results = _load_case(results_dir, "hd_oscillating", heads, d)

    half = max(1, len(ou_results[heads[0]]["G2"]) // 2)

    # =========================================================================
    # 2 × 3 combined figure: rows = OU / Oscillating
    #                         cols = G² time series / H-scaling / EDI
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.subplots_adjust(wspace=0.38, hspace=0.35)

    for row_idx, (results, case_label, stoch, color_case) in enumerate([
        (ou_results, "OU", True, C_CASE2),
        (osc_results, "Oscillating", False, C_CASE1),
    ]):
        # --- Col 0: G² time series ---
        ax_ts = axes[row_idx, 0]
        mean_vals = []
        for H, color in zip(heads, HEAD_COLORS_LOCAL):
            res = results[H]
            g2 = np.maximum(res["G2"], 1e-14)
            ax_ts.semilogy(res["t"], g2, color=color, lw=2.0, label=f"$H={H}$")
            if stoch and "G2_std" in res:
                lo = np.maximum(g2 - res["G2_std"], 1e-14)
                ax_ts.fill_between(res["t"], lo, g2 + res["G2_std"],
                                   color=color, alpha=0.15, linewidth=0)
            mean_vals.append(np.mean(res["G2"][half:]))
        ax_ts.set_xlabel(r"$t$")
        ax_ts.set_ylabel(r"$\mathscr{G}_t^2$")
        ax_ts.grid(True, alpha=0.25, linestyle=":")
        ax_ts.set_title(rf"{case_label} — $\mathscr{{G}}^2$ time series ($d={d}$)")
        ax_ts.legend(fontsize=9)

        # --- Col 1: G² vs H scaling ---
        ax_sc = axes[row_idx, 1]
        H_arr = np.array(heads, dtype=float)
        mean_arr = np.array(mean_vals)
        ax_sc.scatter(H_arr, mean_arr, color="#1a5276", s=60, zorder=5,
                      label=r"$\overline{\mathscr{G}^2}$")
        if len(heads) >= 2:
            try:
                popt, _ = curve_fit(lambda H, a, b: a * H ** b, H_arr, mean_arr)
                H_fit = np.logspace(np.log10(H_arr.min()), np.log10(H_arr.max()), 200)
                ax_sc.plot(H_fit, popt[0] * H_fit ** popt[1],
                           color="#7f8c8d", lw=2.0, ls="--",
                           label=rf"$aH^b$, $b={popt[1]:.3f}$")
            except RuntimeError:
                pass
        if stoch and len(heads) >= 2:
            c_ref = mean_arr[0] * H_arr[0]
            ax_sc.plot(H_arr, c_ref / H_arr, color=color_case, lw=1.5, ls=":",
                       label=r"$\mathcal{O}(H^{-1})$")
        ax_sc.set_xscale("log")
        ax_sc.set_yscale("log")
        ax_sc.set_xlabel(r"$H$")
        ax_sc.set_ylabel(r"$\overline{\mathscr{G}^2}$")
        ax_sc.grid(True, which="both", ls=":", alpha=0.25)
        ax_sc.legend(fontsize=9)
        ax_sc.set_title(rf"{case_label} — $\overline{{\mathscr{{G}}^2}}$ vs $H$ ($d={d}$)")

        # --- Col 2: EDI balance (largest H) ---
        ax_edi = axes[row_idx, 2]
        H_max = max(heads)
        res = results[H_max]
        t = res["t"]
        delta_E = res["E"] - res["E"][0]
        cum_G2 = res["G2_int"]
        cum_dE = res["dE_int"]

        if stoch:
            cum_add = res["add_int"]
            lhs = delta_E + cum_G2 - cum_dE - cum_add
            if "E_std" in res:
                _shade(ax_edi, t, delta_E, res["E_std"], C_ENG)
            if "G2_int_std" in res:
                _shade(ax_edi, t, cum_G2, res["G2_int_std"], C_DIS)
            if "dE_int_std" in res:
                _shade(ax_edi, t, cum_dE, res["dE_int_std"], C_CASE2)
            if "add_int_std" in res:
                _shade(ax_edi, t, cum_add, res["add_int_std"], "pink")
            ax_edi.plot(t, cum_add, color="pink", lw=2.0, ls="-.", label=r"$M_t$")
        else:
            lhs = delta_E + cum_G2 - cum_dE

        ax_edi.plot(t, delta_E, color=C_ENG, lw=2.2, label=r"$\Delta\mathcal{E}$")
        ax_edi.plot(t, cum_G2, color=C_DIS, lw=2.2, ls="--",
                    label=r"$\int_0^t\mathscr{G}_s^2\,ds$")
        ax_edi.plot(t, cum_dE, color=C_CASE2, lw=2.0, ls=":",
                    label=r"$\int_0^t\partial_s\mathcal{E}\,ds$")
        ax_edi.plot(t, lhs, color=C_SUM, lw=3.0, label=r"EDI ($\approx 0$)")
        ax_edi.axhline(0, color="#2c3e50", lw=1.4, alpha=0.6)
        ax_edi.set_xlabel(r"$t$")
        ax_edi.grid(True, which="both", alpha=0.25, linestyle=":")
        ax_edi.set_title(rf"{case_label} EDI ($H={H_max}$, $d={d}$)")
        ax_edi.legend(fontsize=8, ncol=2)

    fig.suptitle(
        rf"High-dimensional experiments on $S^{{{d-1}}}$: OU vs Oscillating",
        fontsize=14, fontweight="bold",
    )
    out_path = os.path.join(figures_dir, f"hd_combined_d{d}.pdf")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  [hd_combined_plots] saved {out_path}")
