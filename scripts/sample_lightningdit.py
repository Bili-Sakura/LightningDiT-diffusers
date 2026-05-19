#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import torch

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from diffusers import LightningDiTPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Sample images with a converted LightningDiT Diffusers pipeline.")
    parser.add_argument("--model", required=True, help="Path to a converted LightningDiT pipeline directory.")
    parser.add_argument("--class-label", type=int, action="append", required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-inference-steps", type=int, default=250)
    parser.add_argument("--guidance-scale", type=float, default=6.7)
    parser.add_argument("--guidance-low", type=float, default=0.0)
    parser.add_argument("--guidance-high", type=float, default=1.0)
    parser.add_argument("--cfg-interval-start", type=float, default=0.125)
    parser.add_argument("--timestep-shift", type=float, default=0.3)
    parser.add_argument("--heun", action="store_true")
    parser.add_argument("--latent-stats", default=None, help="Path to latents_stats.pt (channel mean/std).")
    parser.add_argument("--latent-multiplier", type=float, default=1.0)
    parser.add_argument("--torch-dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default="samples")
    return parser.parse_args()


def _load_latent_stats(path: str):
    stats = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(stats, dict):
        return stats.get("mean"), stats.get("std")
    raise ValueError("latent-stats file must contain 'mean' and 'std' tensors.")


def main():
    args = parse_args()
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.torch_dtype]
    generator_device = args.device if args.device != "cpu" and torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=generator_device)
    if args.seed is not None:
        generator.manual_seed(args.seed)

    latent_mean = latent_std = None
    if args.latent_stats:
        latent_mean, latent_std = _load_latent_stats(args.latent_stats)

    pipe = LightningDiTPipeline.from_pretrained(args.model, torch_dtype=dtype).to(args.device)
    output = pipe(
        class_labels=args.class_label,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        guidance_interval=(args.guidance_low, args.guidance_high),
        cfg_interval_start=args.cfg_interval_start,
        timestep_shift=args.timestep_shift,
        heun=args.heun,
        latent_mean=latent_mean,
        latent_std=latent_std,
        latent_multiplier=args.latent_multiplier,
        generator=generator,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(output.images):
        image.save(output_dir / f"{index:06d}.png")


if __name__ == "__main__":
    main()
