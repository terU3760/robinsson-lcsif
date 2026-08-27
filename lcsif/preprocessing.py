"""Create LCSIF JSON Lines files from standard four-column trajectory files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .social_circle import compute_social_circle


def load_tracks(directory: Path):
    rows = []
    for path in sorted(directory.glob("*.txt")):
        data = np.loadtxt(path)
        if data.ndim == 1:
            data = data[None]
        rows.append((path.stem, data[:, :4]))
    if not rows:
        raise FileNotFoundError(f"No trajectory .txt files found in {directory}")
    return rows


def make_examples(scene, data, obs_len, pred_len, sectors):
    frames = np.unique(data[:, 0])
    length = obs_len + pred_len
    for offset in range(0, len(frames) - length + 1):
        selected_frames = frames[offset:offset + length]
        window = data[np.isin(data[:, 0], selected_frames)]
        for ped_id in np.unique(window[:, 1]):
            track = window[window[:, 1] == ped_id]
            if len(track) != length:
                continue
            trajectory = track[np.argsort(track[:, 0]), 2:4]
            observed = trajectory[:obs_len]
            neighbor_tracks = []
            for other_id in np.unique(window[:, 1]):
                if other_id == ped_id:
                    continue
                other = window[window[:, 1] == other_id]
                other = other[np.isin(other[:, 0], selected_frames[:obs_len])]
                if len(other) == obs_len:
                    neighbor_tracks.append(other[np.argsort(other[:, 0]), 2:4])
            neighbors = np.asarray(neighbor_tracks, dtype=np.float32)
            prompt = f"question: What trajectory does pedestrian {int(ped_id)} follow for the next {pred_len} frames? context: Pedestrian {int(ped_id)} moved along the trajectory {np.round(observed, 2).tolist()} for {obs_len} frames. answer:"
            yield {
                "scene": scene, "frame": int(selected_frames[0]), "ped_id": int(ped_id),
                "obs_traj": observed.tolist(), "pred_traj": trajectory[obs_len:].tolist(),
                "social_circle": compute_social_circle(observed, neighbors, sectors).tolist(),
                "observation": prompt,
                "forecast": f"Pedestrian {int(ped_id)} will move along the trajectory {np.round(trajectory[obs_len:], 2).tolist()} for the next {pred_len} frames.",
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--obs-len", type=int, default=8)
    parser.add_argument("--pred-len", type=int, default=12)
    parser.add_argument("--metric", default="meter")
    parser.add_argument("--sectors", type=int, default=8)
    parser.add_argument("--multimodal", action="store_true")
    args = parser.parse_args()
    source = args.input_root / args.dataset / args.split
    suffix = "-multimodal" if args.multimodal else ""
    destination = args.output_root / "preprocessed" / f"{args.dataset}-{args.split}-{args.obs_len}-{args.pred_len}-{args.metric}{suffix}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for scene, data in load_tracks(source):
            for example in make_examples(scene, data, args.obs_len, args.pred_len, args.sectors):
                stream.write(json.dumps(example, separators=(",", ":")) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
