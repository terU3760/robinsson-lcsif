"""JSON Lines trajectory dataset and tokenizer collation."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    def __init__(self, path: str | Path):
        with Path(path).open(encoding="utf-8") as stream:
            self.rows = [json.loads(line) for line in stream if line.strip()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class Collator:
    def __init__(self, tokenizer, max_length: int, require_social_circle: bool):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.require_social_circle = require_social_circle

    def __call__(self, rows):
        encoded = self.tokenizer(
            [row["observation"] for row in rows], padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        batch = {
            **encoded,
            "pred_traj": torch.tensor([row["pred_traj"] for row in rows], dtype=torch.float32),
        }
        if self.require_social_circle:
            batch["social_circle"] = torch.tensor(
                [row["social_circle"] for row in rows], dtype=torch.float32
            )
        return batch

