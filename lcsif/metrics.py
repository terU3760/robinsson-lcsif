"""Trajectory displacement metrics."""

import torch


def ade_fde(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    distance = torch.linalg.vector_norm(prediction - target, dim=-1)
    return distance.mean(), distance[:, -1].mean()

