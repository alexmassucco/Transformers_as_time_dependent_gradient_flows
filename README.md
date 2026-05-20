# Transformers as time-dependent gradient flows

[![arXiv](https://img.shields.io/badge/arXiv-2605.18870-b31b1b.svg)](https://arxiv.org/abs/2605.18870)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Official simulation and experiment code for:**
> *Multi-Headed Transformer Architectures as Time-dependent Wasserstein Gradient Flows*
> Alex Massucco, Leonardo Del Grande, Marcello Carioni, Christoph Brune, Carola-Bibiane Schönlieb
> [arXiv:2605.18870](https://arxiv.org/abs/2605.18870)

---

## Paper summary

This paper closes the gap between the autonomous mean-field transformer analysis of prior work and the genuinely **non-autonomous, multi-head architectures** used in practice. The key contributions are:

- **Transformer PDE**: token distributions evolve as a Wasserstein gradient flow of a time-dependent interaction energy, with the weight distribution Θ_t entering explicitly.
- **Weighted Wasserstein geometry**: an effective scalar mobility b^{Θ_t}_μ(x) — the Θ_t-weighted harmonic mean of head mobilities — governs the transport geometry and yields an energy-dissipation identity (EDI).
- **Long-time behaviour**: under an integrability condition on Θ_t (satisfied by OU weights, violated by oscillating weights), the ω-limit set consists of stationary points and G² decays at rate O(H⁻¹) in the number of heads (Theorem 4.16).
- **Stability**: a Grönwall-type estimate for initial-data perturbations, and Γ-convergence for weight perturbations.
- **Numerical validation** (Section 6): Monte Carlo simulations on S² confirm the EDI and the O(H⁻¹) scaling in the OU regime; oscillating weights exhibit no decay, verifying the sharpness of the integrability assumption.

---

## Overview

This repository implements the numerical experiments from Section 6 of the paper. It simulates the mean-field dynamics of multi-head self-attention as a **time-dependent gradient flow on the sphere** S^{d-1}.

Tokens are modelled as particles constrained to S^{d-1}; their motion is driven by attention-induced drift and projected back to the tangent space to preserve geometry. The code tracks energy, metric slopes, dissipation, and stochastic corrections, and produces consistent diagnostics and visualisations across S¹, S², and higher-dimensional spheres.

## Simulated system (Equation 5)

The simulations implement the coupled system from Equation (5) in the paper. Token positions x_i(t) on the sphere evolve by a projected gradient flow of an attention energy E, while the attention head weights D_h(t) evolve in time:

$$
\dot{x}_i(t) = -P_{x_i(t)} \nabla_{x_i} \mathcal{E}(x(t); D(t)), \qquad
dD_h(t) = f_h(D_h(t), t)\,dt + \sigma\, dW_h(t).
$$

Here P_{x_i} is the projection onto the tangent space of the sphere. In practice, f_h is either an oscillating deterministic drift or an Ornstein–Uhlenbeck mean-reverting drift, and the stochastic term σ dW_h injects noise into the weight dynamics.

### Attention energy

$$
\mathcal{E}(x; D) = \frac{1}{2Hn^2} \sum_{h=1}^H \sum_{i,j=1}^n e^{\langle x_i,\, D_h\, x_j\rangle}
$$

### Energy-dissipation identity (EDI) — discrete analogue of Eq. (81)

$$
E_T - E_0 = \underbrace{\sum_k \frac{1}{2Hn^2}\sum_{i,j,h} e^{\langle x_i, D^{(h)}_{t_k} x_j\rangle}(x_i x_j^\top : f(D^{(h)}_{t_k}, t_k))\,\Delta t}_{\int_0^T \partial_t\mathcal{E}\,dt}
+ \underbrace{\sum_k \frac{1}{4Hn^2}\sum_{i,j,h} e^{\langle x_i, D^{(h)}_{t_k} x_j\rangle}(x_i x_j^\top : g(D^{(h)}_{t_k}, t_k))^2\,\Delta t}_{\text{Itô correction}}
- \underbrace{\sum_k \mathscr{G}^2_{t_k}\,\Delta t}_{\int_0^T \mathscr{G}^2\,dt} + M_T
$$

The total balance (EDI ≈ 0) is verified in every experiment as a consistency check.

---

## Repository structure

```
.
├── main.py                         # CLI entry point
├── attn_sphere/                    # Core engine + helpers
│   ├── engine.py                   # Euler(-Maruyama) integrator + metrics
│   ├── drift.py                    # Oscillating/OU drift factories (all d ≥ 2)
│   ├── init_tokens.py              # Token initialisers
│   └── plot_utils.py               # Shared plotting helpers
└── experiments/
    ├── s1_oscillating.py            # S¹ deterministic experiment
    ├── s1_ou.py                     # S¹ OU stochastic experiment
    ├── s2_oscillating.py            # S² deterministic experiment
    ├── s2_ou.py                     # S² OU stochastic experiment
    ├── s2_combined_plots.py         # Combined OU vs oscillating (S²)
    ├── hd_oscillating.py            # S^{d-1} deterministic (d ≥ 4, no point cloud)
    ├── hd_ou.py                     # S^{d-1} OU stochastic (d ≥ 4, no point cloud)
    ├── hd_combined_plots.py         # Combined OU vs oscillating (high-d)
    └── gif_utils.py                 # GIF generation
```

---

## Quickstart

```bash
pip install -r requirements.txt

python main.py list

# S¹ experiments
python main.py run s1_oscillating --cpu --gif
python main.py run s1_ou          --cpu --n-mc 5 --gif

# S² experiments (reproduce paper figures)
python main.py run s2_oscillating --cpu --gif
python main.py run s2_ou          --cpu --n-mc 100
python main.py run s2_combined_plots --cpu

# High-dimensional experiments on S^{d-1} (no point cloud, pure diagnostics)
python main.py run hd_oscillating    --cpu --d 5
python main.py run hd_ou             --cpu --d 5 --n-mc 20
python main.py run hd_combined_plots --cpu --d 5
```

Results are saved to `./results/` (raw `.pt` tensors) and figures/GIFs to `./figures/`.

---

## Key diagnostics

The simulations expose a few core objects that are tracked over time and are the main objects of theoretical interest:

1. **Energy E.** A scalar potential summarising how aligned tokens are under the attention kernel. The projected gradient flow tends to decrease E; with noise it can fluctuate.

2. **Metric slope / strong upper gradient G².** The size of the projected gradient (tracked as G²). In the OU regime, Theorem 4.16 predicts G² → 0 at rate O(H⁻¹); in the oscillating regime, it stays bounded away from zero.

3. **Energy-dissipation identity (EDI).** Links ΔE(t) to the time-integrated metric slope and the stochastic correction M_t:
   ```
   ΔE(t) = ∫₀ᵗ ∂_s E ds  −  ∫₀ᵗ G_s² ds  +  M_t
   ```
   The residual (EDI ≈ 0) is a numerical consistency check for the gradient-flow structure.

4. **G² vs H scaling.** Log-log fit of time-averaged G² against H. OU weights should give exponent b ≈ −1 (O(H⁻¹)), while oscillating weights give b ≈ 0 (no decay).

5. **Stochastic correction M_t.** In OU runs, the noise contributes an Itô correction term. This reflects how randomness alters energy accounting.

---

## Experiments

| Name | Sphere | Drift | Produces |
|------|--------|-------|---------|
| `s1_oscillating` | S¹ | deterministic rotating | snapshots, G², EDI, GIF |
| `s1_ou` | S¹ | OU SDE with noise | snapshots, G², EDI, GIF |
| `s2_oscillating` | S² | deterministic rotating | snapshots, G², EDI, GIF |
| `s2_ou` | S² | OU SDE with noise | snapshots, G², EDI, GIF |
| `s2_combined_plots` | S² | compares OU vs oscillating | combined PDFs |
| `hd_oscillating` | S^{d-1} (d≥4) | deterministic rotating | G², EDI, H-scaling |
| `hd_ou` | S^{d-1} (d≥4) | OU SDE with noise | G² ±σ, EDI, H-scaling |
| `hd_combined_plots` | S^{d-1} (d≥4) | compares OU vs oscillating | 2×3 combined PDF |

### What each pair of experiments demonstrates

- **OU (oscillating weights = False)**: satisfies the integrability assumption of Theorem 4.16. Tokens relax to a stationary configuration and G² ∼ O(H⁻¹). Provides quantitative confirmation of the mean-field limit as H → ∞.
- **Oscillating weights**: violates the integrability assumption. The weight matrices orbit a time-periodic target, ∂_t E grows without bound, tokens do not cluster, and G² remains bounded away from zero — illustrating the sharpness of the theory.

---

## CLI reference

```
python main.py run <name(s)|all> [flags]

Flags:
  --cpu               Force CPU (default: use GPU if available)
  --gif               Generate GIF animations (S¹/S² only)
  --no-plot           Skip figure generation
  --T   FLOAT         Final time (default varies by experiment)
  --dt  FLOAT         Time step
  --n   INT           Number of tokens
  --d   INT           Ambient dimension for high-d experiments (default: 5)
  --H   INT [INT ...] Head counts
  --n-mc INT          Monte Carlo paths (OU experiments)
  --noise-var FLOAT   Noise variance σ² (default: 2.0)
  --seed INT          Random seed (default: 42)
  --results-dir PATH  Output directory for .pt files (default: ./results)
  --figures-dir PATH  Output directory for figures (default: ./figures)
```

---

## Citation

If you use this repository, please cite:

```bibtex
@misc{massucco2026multiheadedtransformerarchitecturestimedependent,
      title={Multi-Headed Transformer Architectures as Time-dependent Wasserstein Gradient Flows}, 
      author={Alex Massucco and Leonardo Del Grande and Marcello Carioni and Christoph Brune and Carola-Bibiane Schönlieb},
      year={2026},
      eprint={2605.18870},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.18870}, 
}
```

---

## Notes

- `attn_sphere/` contains the reusable simulation core and plotting helpers.
- `experiments/` provides runnable scripts; all parameters can be overridden from the CLI.
- The oscillating drift (`make_drift_oscillating`) supports any d ≥ 2. For d=2 and d=3 it uses the exact formulas from the paper; for d ≥ 4 it uses a natural generalisation with the same oscillating symmetric structure.
- High-dimensional experiments (`hd_*`) deliberately omit sphere point-cloud plots since the token cloud cannot be visualised in d ≥ 4 — they focus exclusively on the theory-relevant scalar diagnostics.
