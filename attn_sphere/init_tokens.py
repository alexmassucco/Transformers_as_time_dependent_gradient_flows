"""Token initializers and symmetric weight seeds."""

import numpy as np
import torch


def tokens_s1_clustered(n=100, k=5, seed=42, noise=0.05):
    """Points on S¹ clustered around k base angles."""
    rng = np.random.default_rng(seed)
    base_angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
    angles = rng.choice(base_angles, n) + rng.normal(0, noise, n)
    return np.stack([np.cos(angles), np.sin(angles)], axis=-1).astype(np.float32)


def tokens_s2_octahedral(n=200, seed=42, noise=0.05):
    """Points on S² clustered around octahedral vertices."""
    rng = np.random.default_rng(seed)
    verts = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=np.float32,
    )
    pts = verts[rng.choice(len(verts), n)] + rng.normal(0, noise, (n, 3)).astype(
        np.float32
    )
    return pts / np.linalg.norm(pts, axis=-1, keepdims=True)


def symmetric_random_weights(H, d, seed=42, device="cpu", dtype=torch.float64):
    """Random symmetric weight matrices for H heads."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    Z = torch.randn((H, d, d), dtype=dtype, device=device, generator=gen)
    return Z + Z.transpose(-2, -1)
