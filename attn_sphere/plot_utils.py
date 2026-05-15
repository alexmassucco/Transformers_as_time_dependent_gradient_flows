"""Matplotlib helpers shared across experiments."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG = "#fafaf8"
C_CASE1 = "#c0392b"
C_CASE2 = "#27ae60"
C_SUM = "#e67e22"
C_ENG = "#1f77b4"
C_DIS = "#7d3c98"
C_ORIG = "#7f8c8d"
HEAD_COLORS = ["#c0392b", "#e67e22", "#2980b9", "#16a085"]

_U, _V = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 12j]


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def draw_s1_circle(ax, color="#ccc"):
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color=color, lw=0.8)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_s1_scatter(ax, pts, color=C_CASE1, size=28):
    if pts is None:
        ax.text(
            0,
            0,
            "no trajectory",
            ha="center",
            va="center",
            fontsize=8,
            color="#aaa",
        )
        return
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        c=color,
        s=size,
        edgecolors="white",
        linewidths=0.3,
        zorder=5,
    )


def setup_s2_axis(ax, rho=1.0, elev=18, azim=-55):
    ax.plot_wireframe(
        rho * np.cos(_U) * np.sin(_V),
        rho * np.sin(_U) * np.sin(_V),
        rho * np.cos(_V),
        color="#ddd",
        linewidth=0.25,
        alpha=0.6,
    )
    ax.set_xlim(-1.2 * rho, 1.2 * rho)
    ax.set_ylim(-1.2 * rho, 1.2 * rho)
    ax.set_zlim(-1.2 * rho, 1.2 * rho)
    ax.set_box_aspect([1, 1, 1])
    ax.axis("off")
    ax.view_init(elev=elev, azim=azim)


def draw_s2_scatter(ax, pts, color=C_CASE2, size=24):
    if pts is None:
        ax.text(
            0,
            0,
            0,
            "no trajectory",
            ha="center",
            va="center",
            fontsize=8,
            color="#aaa",
        )
        return
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        color=color,
        s=size,
        depthshade=False,
        edgecolors="white",
        linewidths=0.3,
        zorder=5,
    )


def plot_energy_balance(ax, res, title, stoch=False):
    t = res["t"]
    delta_E = res["E"] - res["E"][0]
    cum_G2 = res["G2_int"]
    cum_dE = res["dE_int"]

    if stoch:
        cum_add = res["add_int"]
        lhs = delta_E + cum_G2 - cum_dE - cum_add
        if "E_std" in res:
            ax.fill_between(t, delta_E - res["E_std"], delta_E + res["E_std"], color=C_ENG, alpha=0.15)
        if "G2_int_std" in res:
            ax.fill_between(t, cum_G2 - res["G2_int_std"], cum_G2 + res["G2_int_std"], color=C_DIS, alpha=0.15)
        if "dE_int_std" in res:
            ax.fill_between(t, cum_dE - res["dE_int_std"], cum_dE + res["dE_int_std"], color=C_CASE2, alpha=0.12)
        if "add_int_std" in res:
            ax.fill_between(t, cum_add - res["add_int_std"], cum_add + res["add_int_std"], color="pink", alpha=0.15)
        ax.plot(t, cum_add, color="pink", lw=2.0, ls="-.", label=r"$M_t$")
    else:
        lhs = delta_E + cum_G2 - cum_dE

    ax.plot(t, delta_E, color=C_ENG, lw=2.2, label=r"$\Delta \mathcal{E}$")
    ax.plot(t, cum_G2, color=C_DIS, lw=2.2, ls="--", label=r"$\int_0^t \mathscr{G}_s^2 ds$")
    ax.plot(t, cum_dE, color=C_CASE2, lw=2.0, ls=":", label=r"$\int_0^t \partial_s \mathcal{E}\,ds$")
    ax.plot(t, lhs, color=C_SUM, lw=3.0, label=r"EDI ($\approx 0$)")
    ax.axhline(0, color="#2c3e50", lw=1.4, alpha=0.6)
    ax.set_xlabel(r"$t$")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, which="both", alpha=0.25, linestyle=":")
