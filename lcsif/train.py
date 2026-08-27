"""Accelerate-based training and evaluation entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import DDPMScheduler, LMSDiscreteScheduler, UNet2DConditionModel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_scheduler

from .config import load_config
from .data import Collator, TrajectoryDataset
from .metrics import ade_fde
from .model import AssembledModel


def build_model(config):
    name = config["model"]["text_encoder"]
    text_encoder = AutoModelForSeq2SeqLM.from_pretrained(name).get_encoder()
    if config["training"]["freeze_text_encoder"]:
        text_encoder.requires_grad_(False)
    condition_dim = config["model"]["condition_dim"]
    unet = UNet2DConditionModel(
        sample_size=(1, config["data"]["pred_len"]), in_channels=2, out_channels=2,
        layers_per_block=2, block_out_channels=(128, 256, 256),
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=condition_dim, attention_head_dim=8,
    )
    diffusion = config["diffusion"]
    scheduler_class = LMSDiscreteScheduler if diffusion["scheduler"] == "lms" else DDPMScheduler
    scheduler = scheduler_class(
        beta_start=diffusion["beta_start"], beta_end=diffusion["beta_end"],
        beta_schedule=diffusion["beta_schedule"],
        num_train_timesteps=diffusion["train_timesteps"],
    )
    return AssembledModel(text_encoder, unet, scheduler, config)


def build_loaders(config, tokenizer):
    root = Path(config["data"]["root"]) / "preprocessed"
    dataset = config["data"]["dataset"]
    stem = f"{dataset}-{{split}}-{config['data']['obs_len']}-{config['data']['pred_len']}-{config['data']['metric']}"
    train_path = root / (stem.format(split="train") + "-multimodal.json")
    val_path = root / (stem.format(split="val") + ".json")
    collator = Collator(tokenizer, config["model"]["max_text_length"], config["social_circle"]["enabled"])
    kwargs = {"collate_fn": collator, "num_workers": config["data"]["workers"]}
    train = DataLoader(TrajectoryDataset(train_path), batch_size=config["training"]["batch_size"], shuffle=True, **kwargs)
    val = DataLoader(TrajectoryDataset(val_path), batch_size=config["evaluation"]["batch_size"], **kwargs)
    return train, val


@torch.no_grad()
def evaluate(model, loader, accelerator, config):
    model.eval()
    predictions, targets = [], []
    unwrapped = accelerator.unwrap_model(model)
    for batch_index, batch in enumerate(loader):
        samples = []
        for sample_index in range(config["evaluation"]["samples"]):
            generator = torch.Generator(device=accelerator.device)
            generator.manual_seed(config["evaluation"]["seed"] + batch_index * 1000 + sample_index)
            unwrapped.eval_generator = generator
            samples.append(model(**batch))
        stacked = torch.stack(samples)
        mode = config["evaluation"]["mode"]
        if mode in {"deterministic_mean", "mean_of_k"}:
            prediction = stacked.mean(0)
        elif mode == "best_of_k":
            target = batch["pred_traj"]
            errors = torch.linalg.vector_norm(stacked - target[None], dim=-1).mean(-1)
            best = errors.argmin(0)
            prediction = stacked[best, torch.arange(stacked.shape[1], device=best.device)]
        else:
            prediction = stacked[0]
        predictions.append(accelerator.gather_for_metrics(prediction))
        targets.append(accelerator.gather_for_metrics(batch["pred_traj"]))
    unwrapped.eval_generator = None
    return ade_fde(torch.cat(predictions), torch.cat(targets))


def run(config):
    output = Path(config["training"]["output_dir"])
    accelerator = Accelerator(
        gradient_accumulation_steps=config["training"]["gradient_accumulation"],
        mixed_precision=config["training"]["mixed_precision"],
        log_with="tensorboard", project_dir=output,
    )
    set_seed(config["training"]["seed"])
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["tokenizer"])
    model = build_model(config)
    if config["training"].get("resume"):
        state = torch.load(config["training"]["resume"], map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    train_loader, val_loader = build_loaders(config, tokenizer)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(parameters, lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    total_steps = len(train_loader) * config["training"]["epochs"] // config["training"]["gradient_accumulation"]
    lr_scheduler = get_scheduler(
        config["training"]["lr_scheduler"], optimizer,
        num_warmup_steps=config["training"]["warmup_steps"], num_training_steps=total_steps,
    )
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )
    output.mkdir(parents=True, exist_ok=True)
    best_ade = float("inf")
    for epoch in range(config["training"]["epochs"]):
        model.train()
        for batch in train_loader:
            with accelerator.accumulate(model):
                loss = model(**batch)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(parameters, config["training"]["max_grad_norm"])
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
        ade, fde = evaluate(model, val_loader, accelerator, config)
        if accelerator.is_main_process:
            metrics = {"epoch": epoch + 1, "ade": ade.item(), "fde": fde.item()}
            print(json.dumps(metrics))
            if metrics["ade"] < best_ade:
                best_ade = metrics["ade"]
                accelerator.save(accelerator.unwrap_model(model).state_dict(), output / "best_model.pt")
    accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.dataset:
        config["data"]["dataset"] = args.dataset
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
    if args.resume:
        config["training"]["resume"] = str(args.resume)
    run(config)


if __name__ == "__main__":
    main()
