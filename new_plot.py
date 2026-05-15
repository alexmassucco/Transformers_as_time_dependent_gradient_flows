"""
plot_combined.py
================
Adapted from plot_paper.py.
Produces 3 combined figures for S^2 only, pairing Case 2 (OU) and Case 1 (oscillating):

  fig_combined_snapshots.pdf
      Row 0: Case 2 (OU, H=100)         on S^2  -- t=0, t=8,9,10
      Row 1: Case 1 (oscillating, H=1)  on S^2  -- t=0, t=8,9,10
      Dashed vertical separator between t=0 and evolved snapshots.

  fig_combined_energy_balance.pdf
      Left:  Case 2 (OU, H=100)        -- S^2 energy identity
      Right: Case 1 (oscillating, H=1) -- S^2 energy identity
      Single shared legend above.

  fig_combined_gradient.pdf
      Col 0: Case 2 -- semilog G^2 time-series (H=1,10,100) on S^2
      Col 1: Case 2 -- log-log mean-G^2 vs H on S^2
      Col 2: Case 1 -- semilog G^2 time-series (H=1) on S^2
      Col 3: Case 1 -- windowed integral of G^2 on S^2
      Single shared legend above.

Run AFTER simulate.py has completed.
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

RESULTS_DIR = '/store/CIA/am3270/Projects/TransformerDynamics/results/'
root        = './Final_plots/'
os.makedirs(root, exist_ok=True)

plt.rcParams.update({
    'font.family':       'serif',
    'mathtext.fontset':  'cm',
    'axes.spines.top':   False,
    'axes.spines.right': False,
})
BG      = '#fafaf8'
C_CASE1 = '#c0392b'
C_CASE2 = '#27ae60'
C_SUM   = '#e67e22'
C_ENG   = '#1f77b4'
C_DIS   = '#7d3c98'
C_ORIG  = '#7f8c8d'

head_counts  = [1, 10, 100]
colors_heads = ['#c0392b', '#e67e22', '#2980b9']

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print('Loading results...')
case1_results = {H: torch.load(RESULTS_DIR + f'case1_H{H}.pt', weights_only=False)
                 for H in head_counts}
case2_results = {H: torch.load(RESULTS_DIR + f'case2_H{H}.pt', weights_only=False)
                 for H in head_counts}

r1_s2       = case1_results[100]['s2']
x0_S2       = r1_s2['traj'][0]
res_h100_s2 = case2_results[100]['s2']

n_mc = case2_results[100].get('n_mc', '?')
print(f'All results loaded  (case2 n_mc={n_mc})')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def model(H, a, b):
    return a / H + b

def get_mean_at_time(res, t_target):
    if 'mean_traj' not in res:
        return None
    idx = np.argmin(np.abs(res['t'] - t_target))
    mu  = res['mean_traj'][idx]
    return mu / (np.linalg.norm(mu, axis=-1, keepdims=True) + 1e-14)

def cumulative_trapz(y, x):
    dt   = np.diff(x)
    intg = np.zeros(len(x))
    intg[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * dt)
    return intg

def get_windowed_integral(t, y, window=1.0):
    cum_y   = cumulative_trapz(y, t)
    dt_mean = np.mean(np.diff(t))
    steps   = int(round(window / dt_mean))
    if steps == 0 or steps >= len(t):
        return t, np.zeros_like(t)
    return t[:-steps], cum_y[steps:] - cum_y[:-steps]

def compute_empirical_asymptote(x_np, M_samples=10_000):
    x        = torch.tensor(x_np, dtype=torch.float32)
    n, d     = x.shape
    Ds       = torch.randn(M_samples, d, d)
    Dx       = torch.einsum('nd,Mmd->Mnm', x, Ds)
    logits   = torch.clamp(torch.einsum('id,Mjd->Mij', x, Dx), -50, 50)
    A_unnorm = torch.exp(logits)
    A_norm   = A_unnorm / torch.clamp(A_unnorm.sum(dim=2, keepdim=True), min=1e-14)
    v_unproj = A_norm @ Dx
    v_h      = v_unproj - (v_unproj * x.unsqueeze(0)).sum(dim=-1, keepdim=True) * x.unsqueeze(0)
    v_bar    = v_h.mean(dim=0)
    return (v_bar ** 2).sum(dim=-1).mean().item()

def shade(ax, t, mean, std, color, alpha=0.15):
    ax.fill_between(t, mean - std, mean + std, color=color, alpha=alpha, linewidth=0)

def draw_s2_mean(ax, mu, color, rho=1.0):
    if mu is None:
        ax.text(0, 0, 0, 'mean_traj\nnot stored', ha='center', va='center',
                fontsize=8, color='#aaa', style='italic')
        return
    ax.scatter(rho*mu[:, 0], rho*mu[:, 1], rho*mu[:, 2], color=color, s=25,
               depthshade=False, edgecolors='white', linewidths=0.4, zorder=5)

u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:12j]
snap_times = [16, 18, 20]

def _s2_ax(fig, gs_pos, rho=1.0):
    ax = fig.add_subplot(gs_pos, projection='3d')
    # ax.set_facecolor(BG)
    ax.plot_wireframe(rho*np.cos(u)*np.sin(v), rho*np.sin(u)*np.sin(v), rho*np.cos(v),
                      color='#ccc', linewidth=0.25, alpha=0.5)
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_zlim(-1.2, 1.2)
    ax.set_box_aspect([1, 1, 1]); ax.axis('off')
    ax.view_init(elev=18, azim=-55)
    return ax

# ===========================================================================
# FIG 1 -- fig_combined_snapshots
# ===========================================================================
print('Generating fig_combined_snapshots...')

fig = plt.figure(figsize=(18, 10))
# fig.patch.set_facecolor(BG)

# Use 2 rows × 4 cols, but column 0 will be spanned by both rows for the left panel
gs = fig.add_gridspec(2, 4, wspace=-0.1, hspace=0.1,
                      width_ratios=[.8, .75, .75, .75],
                      left=0.01, right=0.99, top=0.8, bottom=0.04)

row_configs = [
    (res_h100_s2, C_CASE2, r'OU ($H=100$)'),
    (r1_s2,       C_CASE1, r'Oscillating ($H=100$)'),
]

rho0 = .8
rho = 1.

# --- Left panel: span BOTH rows in column 0, centred vertically ---
ax = _s2_ax(fig, gs[:, 0], rho=rho0)
ax.scatter(rho0*x0_S2[:, 0], rho0*x0_S2[:, 1], rho0*x0_S2[:, 2],
           color=C_ORIG, s=55, depthshade=False, edgecolors='white', linewidths=0.5)

# Nudge the left panel further to the left
pos = ax.get_position()
SHIFT = 0.1  # tune this value (figure-fraction units)
ax.set_position([pos.x0 - SHIFT, pos.y0, pos.width, pos.height])

for row_idx, (res, color, row_lbl) in enumerate(row_configs):
    fig.text(.18, 0.61 - row_idx * 0.4, row_lbl, fontsize=16, va='center',
             rotation=0, fontweight='bold', color='#444')

    for col, ts in enumerate(snap_times):
        ax = _s2_ax(fig, gs[row_idx, col + 1], rho)
        draw_s2_mean(ax, get_mean_at_time(res, ts), color, rho)
        if row_idx == 0:
            ax.set_title(f'$t={ts}$', fontsize=18, fontweight='bold', color='#444', pad=0)

# Dashed separator: shift left to stay between the nudged left panel and column 1
fig.add_artist(Line2D([0.15, 0.15], [0.1, 0.8], transform=fig.transFigure,
                      color='#aaaaaa', linestyle='--', lw=1.8, zorder=10))

plt.savefig(root + 'fig_combined_snapshots.pdf', dpi=180, bbox_inches='tight')
plt.close(fig)
print('fig_combined_snapshots saved')

# ===========================================================================
# FIG 3 -- fig_combined_gradient   [LEGEND ABOVE]
# ===========================================================================
print('Generating fig_combined_gradient...')

from scipy.optimize import curve_fit
from scipy.stats import pearsonr

def model(H, a, b):
    return a * H**(b)

def model2(H, a, b):
    return a * np.ones_like(H)

fig, axes = plt.subplots(2, 2, figsize=(22, 18))
# fig.patch.set_facecolor(BG)
fig.subplots_adjust(top=0.85, bottom=0.07, wspace=0.38, hspace=0.25)

Y_LO, Y_HI = 1e-3, 1e1

# ---- Row 0, Col 0: Case 2 semilog ----
ax_ts2 = axes[0, 0]
# ax_ts2.set_facecolor(BG)
mean_vals = []
for H, color in zip(head_counts, colors_heads):
    res  = case2_results[H]['s2']
    t_h  = res['t']
    G2_h = res['G2']
    ax_ts2.semilogy(t_h, np.maximum(G2_h, 1e-14),
                    color=color, lw=2.0, alpha=0.9, label=f'$H={H}$')
    mean_vals.append(np.mean(G2_h[100:]))
# ── NEW: plot e^{-t} for t in [0, 2] ──
# t_exp = np.linspace(0, 2, 300)
# ax_ts2.semilogy(t_exp, np.exp(-t_exp), color='#2c3e50', lw=2.0,
#                 ls=':', alpha=0.85, label=r'$e^{-t}$')
ax_ts2.set_ylim(Y_LO, Y_HI)
ax_ts2.set_xlabel(r'$t$', fontsize=12)
ax_ts2.set_ylabel(r'$\mathscr{G}_t^2$', fontsize=12)
ax_ts2.grid(True, alpha=0.25, linestyle=':')

# ---- Row 0, Col 1: Case 2 log-log ----
ax_vd2 = axes[0, 1]
# ax_vd2.set_facecolor(BG)
all_heads = np.array(head_counts, dtype=float)
all_means = np.array(mean_vals)

popt2, _ = curve_fit(model, all_heads, all_means)
H_fit2    = np.logspace(np.log10(all_heads.min()), np.log10(all_heads.max()), 200)
fit_line2 = model(H_fit2, *popt2)

r2, p2 = pearsonr(1.0 / all_heads, all_means)

ax_vd2.scatter(all_heads, all_means, color='#1a5276', s=70, zorder=5)
ax_vd2.plot(H_fit2, fit_line2, color='#7f8c8d', lw=2.0, ls='--', zorder=0,
            label=f'$a={popt2[0]:.3g}$\n$b={popt2[1]:.3g}$')
# ax_vd2.annotate(
#     rf'$r={r2:.5f}$',
#     xy=(0.97, 0.08), xycoords='axes fraction',
#     ha='right', va='bottom', fontsize=10, color='#7f8c8d',
#     bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.7)
# )
ax_vd2.set_xscale('log'); ax_vd2.set_yscale('log')
ax_vd2.set_ylim(Y_LO, Y_HI)
ax_vd2.set_xlabel(r'$H$', fontsize=12)
ax_vd2.set_ylabel(r'$\overline{\mathscr{G}^2}$', fontsize=12)
ax_vd2.grid(True, which='both', ls=':', alpha=0.25, color='#bbb')
# ── NEW: legend with best-fit curve inside the axes ──
ax_vd2.legend(fontsize=14, framealpha=0.9, loc='best')

print(f"Case 2 (OU)  — a={popt2[0]:.4g}, b={popt2[1]:.4g}, r={r2:.5f}, p={p2:.3e}")
a_ou = popt2[0]

# ---- Row 1, Col 0: Case 1 semilog ----
ax_win = axes[1, 0]
# ax_win.set_facecolor(BG)
mean_vals_case_1 = []
for H, color in zip(head_counts, colors_heads):
    res  = case1_results[H]['s2']
    t_h  = res['t']
    G2_h = res['G2']
    ax_win.semilogy(t_h, np.maximum(G2_h, 1e-14),
                    color=color, lw=2.0, alpha=0.9, label=f'$H={H}$')
    mean_vals_case_1.append(np.mean(G2_h[100:]))
# ── NEW: plot e^{-t} for t in [0, 2] ──
# ax_win.semilogy(t_exp, np.exp(-t_exp), color='#2c3e50', lw=2.0,
#                 ls=':', alpha=0.85, label=r'$e^{-t}$')
ax_win.set_ylim(Y_LO, Y_HI)
ax_win.set_xlabel(r'$t$', fontsize=12)
ax_win.set_ylabel(r'$\mathscr{G}_s^2$', fontsize=12)
ax_win.grid(True, alpha=0.25, linestyle=':')

# ---- Row 1, Col 1: Case 1 log-log ----
ax_ts1 = axes[1, 1]
# ax_ts1.set_facecolor(BG)
all_heads_case_1 = np.array(head_counts, dtype=float)
all_means_case_1 = np.array(mean_vals_case_1)

popt1, _ = curve_fit(model, all_heads_case_1, all_means_case_1)
H_fit1    = np.logspace(np.log10(all_heads_case_1.min()), np.log10(all_heads_case_1.max()), 200)
fit_line1 = model(H_fit1, *popt1)

r1, p1 = pearsonr(1.0 / all_heads_case_1, all_means_case_1)

ax_ts1.scatter(all_heads_case_1, all_means_case_1, color='#1a5276', s=70, zorder=5)
ax_ts1.plot(H_fit1, fit_line1, color='#7f8c8d', lw=2.0, ls='--', zorder=0,
            label=f'$a={popt1[0]:.3g}$\n$b={popt1[1]:.3g}$')
# ax_ts1.annotate(
#     rf'$r={r1:.5f}$',
#     xy=(0.97, 0.5), xycoords='axes fraction',
#     ha='right', va='bottom', fontsize=10, color='#7f8c8d',
#     bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.7)
# )
ax_ts1.set_xscale('log'); ax_ts1.set_yscale('log')
ax_ts1.set_ylim(Y_LO, Y_HI)
ax_ts1.set_xlabel(r'$H$', fontsize=12)
ax_ts1.set_ylabel(r'$\overline{\mathscr{G}^2}$', fontsize=12)
ax_ts1.grid(True, which='both', ls=':', alpha=0.25, color='#bbb')
# ── NEW: legend with best-fit curve inside the axes ──
ax_ts1.legend(fontsize=14, framealpha=0.9, loc='best')

print(f"Case 1 (Osc) — a={popt1[0]:.4g}, b={popt1[1]:.4g}, r={r1:.5f}, p={p1:.3e}")
a_osc = popt1[0]

# ↓ Legend placed ABOVE the subplots (H lines + scatter + O(1/H) reference)
legend_handles = []
for H, color in zip(head_counts, colors_heads):
    legend_handles.append(Line2D([0], [0], color=color, lw=2.0, label=f'$H={H}$'))
legend_handles.append(
    plt.scatter([], [], color='#1a5276', s=60, label=r'$\overline{\mathscr{G}^2}$'))
legend_handles.append(Line2D([0], [0], color='#7f8c8d', lw=2.0, ls='--',
                              label=r'Best-fit $aH^b$'))

fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.94),
           ncol=6, framealpha=0.9, fontsize=18)

pos_row0 = axes[0, 0].get_position().y0 + axes[0, 0].get_position().y1 / 2 - 0.05
pos_row1 = axes[1, 0].get_position().y0 + axes[1, 0].get_position().y1 - 0.03

fig.text(0.5, pos_row0, r'OU ($H=100$)',
         ha='center', va='center', fontsize=16, fontweight='bold',
         transform=fig.transFigure)

fig.text(0.5, pos_row1, r'Oscillating ($H=100$)',
         ha='center', va='center', fontsize=16, fontweight='bold',
         transform=fig.transFigure)

plt.savefig(root + 'fig_combined_gradient.pdf', dpi=180, bbox_inches='tight')
plt.close(fig)
print('fig_combined_gradient saved')


# ===========================================================================
# FIG 2 -- fig_combined_energy_balance   [LEGEND ABOVE]
# ===========================================================================
print('Generating fig_combined_energy_balance...')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
# fig.patch.set_facecolor(BG)
# ↓ top increased to leave headroom for the legend; bottom reduced since legend moved up
fig.subplots_adjust(top=0.78, bottom=0.12, wspace=0.38)

eb_configs = [
    (res_h100_s2, r'OU ($H=100$)',        True),
    (r1_s2,       r'Oscillating ($H=100$)',  False),
]

for ax, (res, title, stoch) in zip(axes, eb_configs):
    # ax.set_facecolor(BG)
    t       = res['t']
    delta_E = res['E'] - res['E'][0]
    cumG2   = res['G2_int']
    cum_dE  = res['dE_int']

    if stoch:
        cum_add = res['add_int']
        LHS = delta_E + cumG2 - cum_dE - cum_add
        if 'E_std'       in res: shade(ax, t,  delta_E, res['E_std'],         C_ENG)
        if 'G2_int_std'  in res: shade(ax, t,  cumG2,   res['G2_int_std'],    C_DIS)
        if 'dE_int_std'  in res: shade(ax, t,  cum_dE,  res['dE_int_std'],    C_CASE2)
        if 'add_int_std' in res: shade(ax, t,  cum_add,  res['add_int_std'],  'pink')
        ax.plot(t, cum_add, color='pink', lw=2.0, ls='-.', label=r'$M_t$')
    else:
        LHS = delta_E + cumG2 - cum_dE

    ax.plot(t,  delta_E, color=C_ENG,   lw=2.2,          label=r'$\Delta \mathcal{E}$')
    ax.plot(t,  cumG2,   color=C_DIS,   lw=2.2, ls='--',  label=r'$\int_0^t \mathscr{G}_s^2\,\mathrm{d}s$')
    ax.plot(t,  cum_dE,  color=C_CASE2, lw=2.0, ls=':',   label=r'$\int_0^t \partial_s \mathcal{E}\,\mathrm{d}s$')
    ax.plot(t,  LHS,     color=C_SUM,   lw=3.0,           label=r'EDI ($\approx 0$)')
    ax.axhline(0, color='#2c3e50', lw=1.5, alpha=0.6)
    shade(ax, t, 0, a_ou*np.ones_like(t)/100, '#2c3e50', alpha=0.15)
    ax.set_xlabel(r'$t$', fontsize=12)
    # ax.set_yscale('symlog', linthresh=1e-3)
    # ax.set_ylabel('Energy terms', fontsize=11)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, which='both', alpha=0.25, linestyle=':')

# ↓ Legend placed ABOVE the subplots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98),
           ncol=5, framealpha=0.9, fontsize=11)

plt.savefig(root + 'fig_combined_energy_balance.pdf', dpi=180, bbox_inches='tight')
plt.close(fig)
print('fig_combined_energy_balance saved')

print('=' * 60)
print('All 3 combined figures saved to:', root)
print('=' * 60)