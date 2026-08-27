"""Configuration loading and validation."""

from __future__ import annotations

import json
from pathlib import Path


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if "extends" in config:
        parent = load_config(path.parent / config.pop("extends"))
        config = _merge(parent, config)
    if config["text_features"]["mode"] not in {"last_token", "full_sequence"}:
        raise ValueError("text_features.mode must be last_token or full_sequence")
    if config["diffusion"]["scheduler"] not in {"lms", "ddpm"}:
        raise ValueError("diffusion.scheduler must be lms or ddpm")
    return config


def _merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        result[key] = _merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result

