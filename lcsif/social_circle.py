"""SocialCircle feature extraction from observed target and neighbor trajectories."""

from __future__ import annotations

import numpy as np


def compute_social_circle(target: np.ndarray, neighbors: np.ndarray, sectors: int = 8) -> np.ndarray:
    """Return per-sector mean speed, distance, and direction for nearby pedestrians."""
    result = np.zeros((sectors, 3), dtype=np.float32)
    if neighbors.size == 0:
        return result
    relative = neighbors[:, -1] - target[-1]
    distance = np.linalg.norm(relative, axis=1)
    velocity = np.linalg.norm(np.diff(neighbors, axis=1), axis=2).mean(axis=1)
    direction = np.mod(np.arctan2(relative[:, 1], relative[:, 0]), 2 * np.pi)
    bins = np.floor(direction / (2 * np.pi / sectors)).astype(int).clip(0, sectors - 1)
    for sector in range(sectors):
        selected = bins == sector
        if selected.any():
            result[sector] = (velocity[selected].mean(), distance[selected].mean(), direction[selected].mean())
    return result

