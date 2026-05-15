# Transformers as time-dependent gradient flows
**Gradient-flow simulations of multi-head self-attention on the sphere, reorganized from `simulate.py` and `new_plot.py`.**

This repository provides a clean, modular layout: a reusable simulation engine in `attn_sphere/`, separate experiments for **S¹** and **S²**, and utilities to generate plots and GIFs of the evolving point clouds.

---

## Structure

```
.
├── main.py                     # CLI entry point
├── attn_sphere/                # Core engine + helpers
│   ├── engine.py               # Euler(-Maruyama) integrator + metrics
│   ├── drift.py                # Oscillating/OU drift factories
│   ├── init_tokens.py          # Token initializers
│   └── plot_utils.py           # Shared plotting helpers
├── experiments/
│   ├── s1_oscillating.py        # S¹ deterministic experiment
│   ├── s1_ou.py                 # S¹ OU stochastic experiment
│   ├── s2_oscillating.py        # S² deterministic experiment
│   ├── s2_ou.py                 # S² OU stochastic experiment
│   ├── s2_combined_plots.py     # Combined OU vs oscillating plots
│   └── gif_utils.py             # GIF generation
├── simulate.py                  # original monolithic script (unchanged)
└── new_plot.py                  # original plotting script (unchanged)
```

---

## Quickstart

```bash
pip install -r requirements.txt

python main.py list

# S¹ / S² examples (CPU)
python main.py run s1_oscillating --cpu --gif
python main.py run s1_ou          --cpu --n-mc 5 --gif
python main.py run s2_oscillating --cpu --gif
python main.py run s2_ou          --cpu --n-mc 5

# Combined plots (requires S² results to exist)
python main.py run s2_combined_plots
```

Results are saved to `./results/` and figures/GIFs to `./figures/`.

---

## Experiments

| Name | Sphere | Drift | Output |
|---|---|---|---|
| `s1_oscillating` | S¹ | deterministic rotating drift | snapshots, G², EDI, GIF |
| `s1_ou` | S¹ | OU SDE with noise | snapshots, G², EDI, GIF |
| `s2_oscillating` | S² | deterministic rotating drift | snapshots, G², EDI, GIF |
| `s2_ou` | S² | OU SDE with noise | snapshots, G², EDI, GIF |
| `s2_combined_plots` | S² | compares OU vs oscillating | combined PDFs |

---

## Mathematical background (short)

Tokens $\mathbf{x}_1,\dots,\mathbf{x}_n \in S^{d-1}$ evolve by the projected gradient flow of the attention energy

$$
\mathcal{E}(x,D)=\frac{1}{2Hn^2}\sum_{h,i,j}e^{x_i^\top D_h x_j}.
$$

The metric slope $\mathscr{G}^2$ and the energy-dissipation identity (EDI) are logged as diagnostics.

---

## Notes

* `simulate.py` and `new_plot.py` are kept verbatim and serve as the original reference.
* The refactor in `attn_sphere/` mirrors the same equations but exposes them as importable functions.
