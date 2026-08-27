"""Convert Stanford Drone Dataset annotations to four-column LCSIF trajectory files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SCENES = ("bookstore", "coupa", "deathCircle", "gates", "hyang", "little", "nexus", "quad")


def read_annotation(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, sep=r"\s+", header=None)
    if data.shape[1] < 10:
        raise ValueError(f"Expected the 10-column SDD annotation format: {path}")
    return pd.DataFrame({
        "frame": data.iloc[:, 5].astype(int), "pedestrian": data.iloc[:, 0].astype(int),
        "x": (data.iloc[:, 1] + data.iloc[:, 3]) / 2,
        "y": (data.iloc[:, 2] + data.iloc[:, 4]) / 2,
        "lost": data.iloc[:, 6].astype(int),
    })


def split_by_time(data: pd.DataFrame, train_ratio: float, val_ratio: float):
    frames = np.sort(data.frame.unique())
    train_end = max(1, int(len(frames) * train_ratio))
    val_end = max(train_end + 1, int(len(frames) * (train_ratio + val_ratio)))
    for name, selected in (("train", frames[:train_end]), ("val", frames[train_end:val_end]), ("test", frames[val_end:])):
        yield name, data[data.frame.isin(selected)]


def convert(root: Path, output: Path, train_ratio: float, val_ratio: float, scale: float):
    for scene in SCENES:
        annotation_files = sorted((root / "annotations" / scene).glob("*/annotations.txt"))
        if not annotation_files:
            annotation_files = sorted((root / "annotation" / scene).glob("*/annotations.txt"))
        for video_index, path in enumerate(annotation_files):
            data = read_annotation(path)
            data = data[data.lost == 0].drop(columns="lost")
            data[["x", "y"]] *= scale
            for split, part in split_by_time(data, train_ratio, val_ratio):
                directory = output / f"sdd_{scene}" / split
                directory.mkdir(parents=True, exist_ok=True)
                values = part[["frame", "pedestrian", "x", "y"]].to_numpy()
                np.savetxt(directory / f"video_{video_index:02d}.txt", values, fmt=("%d", "%d", "%.6f", "%.6f"), delimiter="\t")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdd-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--coordinate-scale", type=float, default=1.0)
    args = parser.parse_args()
    convert(args.sdd_root, args.output_root, args.train_ratio, args.val_ratio, args.coordinate_scale)


if __name__ == "__main__":
    main()

