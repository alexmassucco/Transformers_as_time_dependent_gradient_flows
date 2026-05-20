"""Drift factories for weight dynamics."""

import math
import numpy as np
import torch


def make_drift_oscillating(d):
    """
    Rotating symmetric drift for the deterministic oscillating experiments.
    Supports any d >= 2.  d=2 and d=3 use the exact formulas from the paper;
    d >= 4 uses a natural generalisation with the same structure.
    """

    def drift(D, t):
        # D has shape [P, H, d, d]; H lives at axis -3.
        H = D.shape[-3]
        frames = []
        for h in range(H):
            phi = 2.0 * math.pi * h / max(H, 1)
            tt = t + phi

            if d == 2:
                c = 1.5 * math.cos(tt)
                s2 = math.sin(2.0 * tt)
                frame = [[2.0 + c, s2], [s2, 2.0 - c]]
            elif d == 3:
                c = 1.5 * math.cos(tt)
                s2 = math.sin(2.0 * tt)
                s = math.sin(tt)
                c2 = math.cos(2.0 * tt)
                c_pi4 = 1.5 * math.cos(tt + math.pi / 4.0)
                frame = [
                    [2.0 + c, s2, s],
                    [s2, 2.0 + 1.5 * math.sin(tt), c2],
                    [s, c2, 2.0 + c_pi4],
                ]
            else:
                # General d: symmetric matrix with oscillating diagonal and
                # off-diagonal entries, phase-shifted per head and per index pair.
                n_pairs = max(d * (d - 1) // 2, 1)
                frame = np.zeros((d, d), dtype=np.float64)
                for i in range(d):
                    frame[i, i] = 2.0 + 1.5 * math.cos(tt + 2.0 * math.pi * i / d)
                pair = 0
                for i in range(d):
                    for j in range(i + 1, d):
                        val = math.sin(2.0 * tt + 2.0 * math.pi * pair / n_pairs)
                        frame[i, j] = val
                        frame[j, i] = val
                        pair += 1
                frame = frame.tolist()

            frames.append(frame)

        # target shape: [H, d, d] — broadcasts correctly with D=[P, H, d, d]
        target = D.new_tensor(frames)
        return target - D

    return drift


def ou_drift(D, t):
    """Ornstein-Uhlenbeck mean reversion to the identity."""
    eye = torch.eye(D.shape[-1], device=D.device, dtype=D.dtype)
    return eye.unsqueeze(0).expand_as(D) - D


def frozen_drift(D, t):
    """Frozen weights: dD/dt = 0."""
    return torch.zeros_like(D)
