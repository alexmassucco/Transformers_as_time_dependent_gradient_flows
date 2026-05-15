"""Drift factories for weight dynamics."""

import math
import torch


def make_drift_oscillating(d):
    """
    Rotating symmetric drift used in the deterministic oscillating experiments.
    Supports d=2 and d=3.
    """

    if d not in (2, 3):
        raise ValueError("make_drift_oscillating only supports d=2 or d=3.")

    def drift(D, t):
        H = D.shape[0]
        frames = []
        for h in range(H):
            phi = 2 * math.pi * h / H
            tt = t + phi

            if d == 2:
                c = 1.5 * math.cos(tt)
                s2 = math.sin(2 * tt)
                frame = [[2.0 + c, s2], [s2, 2.0 - c]]
            else:
                c = 1.5 * math.cos(tt)
                s2 = math.sin(2 * tt)
                s = math.sin(tt)
                c2 = math.cos(2 * tt)
                c_pi4 = 1.5 * math.cos(tt + math.pi / 4)
                frame = [
                    [2.0 + c, s2, s],
                    [s2, 2.0 + 1.5 * s, c2],
                    [s, c2, 2.0 + c_pi4],
                ]

            frames.append(frame)

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
