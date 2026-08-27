# ROBINSSON-LCSIF

A unified framework for language-conditioned pedestrian trajectory forecasting with T5-Small, optional SocialCircle features fused through FiLM, and a conditional diffusion UNet.

| Configuration | SocialCircle + FiLM | T5 representation |
|---|---:|---|
| `social-last` | yes | final non-padding token |
| `social-full` | yes | full encoder sequence |
| `text-last` | no | final non-padding token |
| `text-full` | no | full encoder sequence |

## Install

Python 3.10+ and a CUDA-capable PyTorch installation are recommended.

```bash
git clone <repository-url> robinsson-lcsif
cd robinsson-lcsif
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
accelerate config
```

T5-Small is downloaded on first use. For offline operation, set `model.text_encoder` and `model.tokenizer` in `configs/base.json` to a local model directory.

## Prepare ETH/UCY data

Raw files contain whitespace-separated `frame pedestrian_id x y` columns:

```text
raw/eth/train/*.txt
raw/eth/val/*.txt
raw/eth/test/*.txt
```

Generate training-ready JSON Lines files:

```bash
scripts/prepare_dataset.sh eth /path/to/raw /path/to/generated
```

Set `data.root` in `configs/base.json` to `/path/to/generated`. Records contain the prompt, numeric trajectories, and an 8×3 SocialCircle tensor. Text-only ablations use the same files without loading that tensor.

## Stanford Drone Dataset

Download SDD separately, then run:

```bash
lcsif-convert-sdd --sdd-root /path/to/StanfordDroneDataset --output-root raw
scripts/prepare_dataset.sh sdd_bookstore raw data
```

Supported datasets are `sdd_bookstore`, `sdd_coupa`, `sdd_deathCircle`, `sdd_gates`, `sdd_hyang`, `sdd_little`, `sdd_nexus`, and `sdd_quad`. The converter makes chronological 70/15/15 splits. Pass `--coordinate-scale` to the converter when a chosen SDD calibration requires pixel-to-world scaling.

Existing generated experiment files can be used directly by setting `data.root` to the directory that contains `preprocessed/`.

## Train

```bash
scripts/train.sh social-last eth
scripts/train.sh social-full eth
scripts/train.sh text-last eth
scripts/train.sh text-full eth
```

Or launch directly:

```bash
python -m accelerate.commands.launch -m lcsif.train \
  --config configs/social-last.json --dataset sdd_bookstore \
  --output-dir outputs/sdd-bookstore-social-last
```

Shared hyperparameters are in `configs/base.json`; variant files override the ablation axes and the matching cross-attention width (256 after FiLM, 512 for native T5 states). Defaults retain the reference setup: LMS diffusion, learning rate `1.6e-4`, cosine decay, 500 warm-up steps, gradient clipping at 1.0, frozen T5-Small, and deterministic 20-sample validation.

Sampling scales initial noise by the scheduler's `init_noise_sigma`, preserving the corrected sample diversity behavior. Evaluation modes are `deterministic_mean`, `mean_of_k`, `best_of_k`, and `single`. `best_of_k` is an oracle research metric and is not comparable with single-trajectory metrics.

Resume from a state dictionary with `--resume outputs/.../best_model.pt`. The lowest-validation-ADE model is written to `best_model.pt`.

## Verify

```bash
python -m pytest
python -m compileall -q lcsif tests
ruff check .
```

This repository intentionally excludes datasets, weights, logs, caches, debugging variants, collision avoidance, and TrajNet++.

## License

Copyright © 2026 ROBINSSON-LCSIF contributors. GPL-3.0-or-later; see `LICENSE`.
