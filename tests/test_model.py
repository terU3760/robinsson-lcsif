import numpy as np
import torch

from lcsif.metrics import ade_fde
from lcsif.social_circle import compute_social_circle
from lcsif.model import AssembledModel


class Output:
    def __init__(self, value):
        self.last_hidden_state = value
        self.sample = value


class Encoder(torch.nn.Module):
    config = type("Config", (), {"d_model": 16})()

    def forward(self, input_ids, attention_mask):
        return Output(torch.ones(input_ids.shape[0], input_ids.shape[1], 16))


class Unet(torch.nn.Module):
    def forward(self, sample, timestep, encoder_hidden_states):
        return Output(torch.zeros_like(sample))


class Scheduler:
    config = type("Config", (), {"num_train_timesteps": 10})()
    init_noise_sigma = 1.0

    def set_timesteps(self, count, device):
        self.timesteps = torch.arange(count - 1, -1, -1, device=device)

    def add_noise(self, clean, noise, timesteps):
        return clean + noise

    def step(self, predicted, timestep, sample):
        return type("Step", (), {"prev_sample": sample})()


def test_social_circle_shape_and_empty_neighbors():
    target = np.zeros((8, 2), dtype=np.float32)
    result = compute_social_circle(target, np.empty((0, 8, 2), dtype=np.float32))
    assert result.shape == (8, 3)
    assert np.all(result == 0)


def test_displacement_metrics():
    target = torch.zeros(2, 12, 2)
    prediction = torch.ones_like(target)
    ade, fde = ade_fde(prediction, target)
    expected = torch.tensor(2.0).sqrt()
    assert torch.isclose(ade, expected)
    assert torch.isclose(fde, expected)


def test_all_model_ablation_shapes():
    base = {
        "model": {"condition_dim": 8}, "data": {"pred_len": 12},
        "diffusion": {"inference_steps": 2},
        "social_circle": {"sectors": 8, "channels": 4},
    }
    for social in (False, True):
        for mode in ("last_token", "full_sequence"):
            condition_dim = 8 if social else 16
            config = {**base, "model": {"condition_dim": condition_dim},
                      "text_features": {"mode": mode},
                      "social_circle": {**base["social_circle"], "enabled": social}}
            model = AssembledModel(Encoder(), Unet(), Scheduler(), config)
            batch = {
                "input_ids": torch.ones(2, 5, dtype=torch.long),
                "attention_mask": torch.ones(2, 5, dtype=torch.long),
                "pred_traj": torch.zeros(2, 12, 2),
            }
            if social:
                batch["social_circle"] = torch.zeros(2, 8, 3)
            assert model(**batch).ndim == 0
            model.eval()
            assert model(**batch).shape == (2, 12, 2)
