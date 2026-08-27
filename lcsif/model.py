"""Unified language-conditioned diffusion model with optional SocialCircle FiLM fusion."""

from __future__ import annotations

import inspect

import torch
from torch import nn
from torch.nn import functional as F


class FiLM(nn.Module):
    """Feature-wise linear modulation supporting token-wise conditions."""

    def forward(self, features: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        if gamma.ndim == 2:
            gamma, beta = gamma[..., None, None], beta[..., None, None]
            return gamma * features + beta
        gamma, beta = gamma[..., None, None], beta[..., None, None]
        features = features[:, None].expand(-1, gamma.shape[1], -1, -1, -1)
        return gamma * features + beta


class AssembledModel(nn.Module):
    """T5 encoder, optional SocialCircle FiLM fusion, and conditional trajectory UNet."""

    def __init__(self, text_encoder: nn.Module, unet: nn.Module, scheduler, config: dict):
        super().__init__()
        self.text_encoder = text_encoder
        self.unet = unet
        self.scheduler = scheduler
        self.use_social_circle = config["social_circle"]["enabled"]
        self.text_mode = config["text_features"]["mode"]
        self.condition_dim = config["model"]["condition_dim"]
        self.pred_len = config["data"]["pred_len"]
        self.inference_steps = config["diffusion"]["inference_steps"]
        hidden_size = text_encoder.config.d_model
        if self.use_social_circle:
            if hidden_size < 2 * self.condition_dim:
                raise ValueError("T5 hidden size must be at least twice the FiLM condition dimension")
            self.film = FiLM()
        elif hidden_size != self.condition_dim:
            raise ValueError("Text-only condition_dim must equal the T5 hidden size")
        self.eval_generator: torch.Generator | None = None

    def encode_condition(self, input_ids, attention_mask, social_circle=None):
        hidden = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.text_mode == "last_token":
            positions = attention_mask.long().sum(dim=1).sub(1).clamp_min(0)
            hidden = hidden[torch.arange(hidden.shape[0], device=hidden.device), positions][:, None]
        if not self.use_social_circle:
            return hidden
        if social_circle is None:
            raise ValueError("social_circle is required when SocialCircle fusion is enabled")
        gamma, beta = hidden[..., : 2 * self.condition_dim].chunk(2, dim=-1)
        visual = social_circle[:, None].repeat(1, self.condition_dim, 1, 1)
        if hidden.shape[1] == 1:
            fused = self.film(visual, gamma[:, 0], beta[:, 0])
            return fused.permute(0, 2, 3, 1).flatten(1, 2)
        fused = self.film(visual, gamma, beta)
        return fused.permute(0, 1, 3, 4, 2).flatten(1, 3)

    def forward(self, pred_traj, input_ids, attention_mask, social_circle=None):
        condition = self.encode_condition(input_ids, attention_mask, social_circle)
        if self.training:
            clean = pred_traj.transpose(1, 2).unsqueeze(2)
            noise = torch.randn_like(clean)
            self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps, device=clean.device)
            timesteps = torch.randint(
                0, self.scheduler.config.num_train_timesteps, (clean.shape[0],), device=clean.device
            ).long()
            noisy = self.scheduler.add_noise(clean, noise, timesteps)
            predicted = self.unet(noisy, timesteps, encoder_hidden_states=condition).sample
            return F.mse_loss(predicted, noise)
        return self.sample(condition)

    @torch.no_grad()
    def sample(self, condition):
        device = condition.device
        generator = self.eval_generator
        if generator is None:
            generator = torch.Generator(device=device)
            generator.seed()
        self.scheduler.set_timesteps(self.inference_steps, device=device)
        latents = torch.randn(
            condition.shape[0], 2, 1, self.pred_len, device=device, generator=generator
        ) * getattr(self.scheduler, "init_noise_sigma", 1.0)
        step_supports_generator = "generator" in inspect.signature(self.scheduler.step).parameters
        for timestep in self.scheduler.timesteps:
            predicted = self.unet(latents, timestep, encoder_hidden_states=condition).sample
            kwargs = {"generator": generator} if step_supports_generator else {}
            latents = self.scheduler.step(predicted, timestep, latents, **kwargs).prev_sample
        return latents.squeeze(2).transpose(1, 2)
