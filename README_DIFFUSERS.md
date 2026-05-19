LightningDiT Diffusers integration
==================================

This repository mirrors the layout used for upstream Diffusers integration (see [NiT-diffusers](https://github.com/Bili-Sakura/NiT-diffusers.git)). Core code lives under `src/diffusers`:

- `models/transformers/transformer_lightningdit.py` — `LightningDiTTransformer2DModel` (`ModelMixin` / `ConfigMixin`)
- `schedulers/scheduling_flow_match_lightningdit.py` — `LightningDiTFlowMatchScheduler` (flow-matching ODE)
- `pipelines/lightningdit/pipeline_lightningdit.py` — `LightningDiTPipeline` for class-conditional sampling

Convert a checkpoint
--------------------

```bash
pip install -e ".[dev]"

python scripts/convert_lightningdit_to_diffusers.py \
  --checkpoint path/to/lightningdit-xl-imagenet256-800ep.pt \
  --output lightningdit-xl-diffusers \
  --model-size lightningdit-xl/1 \
  --input-size 16 \
  --in-channels 32 \
  --use-swiglu --use-rope --use-rmsnorm \
  --check-load
```

The output directory contains:

```text
model_index.json
scheduler/scheduler_config.json
transformer/config.json
transformer/diffusion_pytorch_model.safetensors
vae_pretrained_model_name_or_path.txt
```

Use `--copy-vae /path/to/vae` to vendor a local VAE into `output/vae`.

Sample from a converted checkpoint
----------------------------------

```bash
python scripts/sample_lightningdit.py \
  --model lightningdit-xl-diffusers \
  --class-label 207 \
  --latent-stats path/to/latents_stats.pt \
  --num-inference-steps 250 \
  --guidance-scale 6.7 \
  --cfg-interval-start 0.125 \
  --timestep-shift 0.3
```

For upstreaming to `huggingface/diffusers`, copy the files under `src/diffusers` into the matching Diffusers package paths and register the classes in Diffusers' lazy import tables.
