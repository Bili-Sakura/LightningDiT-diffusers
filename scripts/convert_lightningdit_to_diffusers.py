#!/usr/bin/env python3
# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import torch

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

try:
    from safetensors.torch import load_file as safe_load_file
    from safetensors.torch import save_file as safe_save_file
except Exception:  # pragma: no cover
    safe_load_file = None
    safe_save_file = None

from diffusers.models.transformers import LightningDiTTransformer2DModel
from diffusers.schedulers import LightningDiTFlowMatchScheduler


MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "lightningdit-b/1": {"depth": 12, "hidden_size": 768, "num_heads": 12, "patch_size": 1},
    "lightningdit-b/2": {"depth": 12, "hidden_size": 768, "num_heads": 12, "patch_size": 2},
    "lightningdit-l/2": {"depth": 24, "hidden_size": 1024, "num_heads": 16, "patch_size": 2},
    "lightningdit-xl/1": {"depth": 28, "hidden_size": 1152, "num_heads": 16, "patch_size": 1},
    "lightningdit-xl/2": {"depth": 28, "hidden_size": 1152, "num_heads": 16, "patch_size": 2},
    "lightningdit-1p0b/1": {"depth": 24, "hidden_size": 1536, "num_heads": 24, "patch_size": 1},
    "lightningdit-1p0b/2": {"depth": 24, "hidden_size": 1536, "num_heads": 24, "patch_size": 2},
    "lightningdit-1p6b/1": {"depth": 28, "hidden_size": 1792, "num_heads": 28, "patch_size": 1},
    "lightningdit-1p6b/2": {"depth": 28, "hidden_size": 1792, "num_heads": 28, "patch_size": 2},
}


def _load_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    if checkpoint_path.endswith(".safetensors"):
        if safe_load_file is None:
            raise ImportError("Install safetensors to convert .safetensors checkpoints.")
        state_dict = safe_load_file(checkpoint_path, device="cpu")
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            for key in ("ema", "state_dict", "model", "module"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break
            if not all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
                state_dict = checkpoint.get("model", checkpoint)
            else:
                state_dict = checkpoint
        else:
            raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    return _clean_state_dict(state_dict)


def _clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    prefixes = ("model.", "module.", "transformer.")
    for key, value in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned[key] = value
    return cleaned


def _save_config(output_dir: Path, config: Dict[str, Any]):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, sort_keys=True)
        file.write("\n")


def _save_weights(output_dir: Path, state_dict: Dict[str, torch.Tensor], safe_serialization: bool):
    output_dir.mkdir(parents=True, exist_ok=True)
    if safe_serialization:
        if safe_save_file is None:
            raise ImportError("Install safetensors or pass --no-safe-serialization.")
        safe_save_file(
            state_dict,
            str(output_dir / "diffusion_pytorch_model.safetensors"),
            metadata={"format": "pt"},
        )
    else:
        torch.save(state_dict, output_dir / "diffusion_pytorch_model.bin")


def _write_model_index(output_dir: Path, vae: str | None):
    model_index = {
        "_class_name": "LightningDiTPipeline",
        "_diffusers_version": "0.30.1",
        "scheduler": ["diffusers", "LightningDiTFlowMatchScheduler"],
        "transformer": ["diffusers", "LightningDiTTransformer2DModel"],
    }
    if vae is not None:
        model_index["vae"] = ["diffusers", "AutoencoderKL"]
    with open(output_dir / "model_index.json", "w", encoding="utf-8") as file:
        json.dump(model_index, file, indent=2, sort_keys=True)
        file.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert original LightningDiT checkpoints to a Diffusers pipeline directory."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a LightningDiT .pt/.bin/.safetensors checkpoint.")
    parser.add_argument("--output", required=True, help="Output Diffusers model directory.")
    parser.add_argument("--model-size", choices=sorted(MODEL_PRESETS), default="lightningdit-xl/1")
    parser.add_argument("--vae", default="hustvl/vavae-imagenet256-f16d32-dinov2")
    parser.add_argument("--copy-vae", default=None, help="Optional local VAE directory to copy into output/vae.")
    parser.add_argument("--input-size", type=int, default=16, help="Latent spatial size (image_size // downsample_ratio).")
    parser.add_argument("--in-channels", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--class-dropout-prob", type=float, default=0.1)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-swiglu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-rope", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-rmsnorm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wo-shift", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--learn-sigma", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--path-type", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--safe-serialization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-load", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    transformer_dir = output_dir / "transformer"
    scheduler_dir = output_dir / "scheduler"

    state_dict = _load_state_dict(args.checkpoint)
    config = {
        "input_size": args.input_size,
        "in_channels": args.in_channels,
        "class_dropout_prob": args.class_dropout_prob,
        "num_classes": args.num_classes,
        "qk_norm": args.qk_norm,
        "use_swiglu": args.use_swiglu,
        "use_rope": args.use_rope,
        "use_rmsnorm": args.use_rmsnorm,
        "wo_shift": args.wo_shift,
        "learn_sigma": args.learn_sigma,
        **MODEL_PRESETS[args.model_size],
    }

    if args.check_load:
        model = LightningDiTTransformer2DModel(**config)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys or unexpected_keys:
            print("Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)
            raise SystemExit(1)

    _save_config(transformer_dir, {"_class_name": "LightningDiTTransformer2DModel", **config})
    _save_weights(transformer_dir, state_dict, args.safe_serialization)

    _save_config(
        scheduler_dir,
        {
            "_class_name": "LightningDiTFlowMatchScheduler",
            "path_type": args.path_type,
            "num_train_timesteps": 1000,
        },
    )

    if args.copy_vae is not None:
        target_vae_dir = output_dir / "vae"
        if target_vae_dir.exists():
            shutil.rmtree(target_vae_dir)
        shutil.copytree(args.copy_vae, target_vae_dir)
    elif args.vae:
        with open(output_dir / "vae_pretrained_model_name_or_path.txt", "w", encoding="utf-8") as file:
            file.write(args.vae + os.linesep)

    _write_model_index(output_dir, args.vae)
    print(f"Saved Diffusers-style LightningDiT pipeline to {output_dir}")


if __name__ == "__main__":
    main()
