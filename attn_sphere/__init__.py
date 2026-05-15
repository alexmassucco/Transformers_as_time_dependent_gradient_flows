"""Top-level API for attention gradient-flow simulations on the sphere."""

from .engine import simulate, simulate_batch
from .drift import make_drift_oscillating, ou_drift, frozen_drift
from .init_tokens import (
    tokens_s1_clustered,
    tokens_s2_octahedral,
    symmetric_random_weights,
)

__all__ = [
    "simulate",
    "simulate_batch",
    "make_drift_oscillating",
    "ou_drift",
    "frozen_drift",
    "tokens_s1_clustered",
    "tokens_s2_octahedral",
    "symmetric_random_weights",
]
