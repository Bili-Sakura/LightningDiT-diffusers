from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput


@dataclass
class LightningDiTPipelineOutput(BaseOutput):
    images: Union[torch.FloatTensor, List]


class LightningDiTPipeline(DiffusionPipeline):
    """
    Class-conditional LightningDiT sampling in latent space with optional VA-VAE decoding.

    The transformer is a LightningDiTTransformer2DModel. ``vae`` may be any module exposing
    ``decode_to_images`` (e.g. tokenizer.vavae.VA_VAE) or None if you only need latents.
    """

    model_cpu_offload_seq = "transformer"
    _optional_components = ["vae"]

    def __init__(self, transformer, vae=None):
        super().__init__()
        self.register_modules(transformer=transformer, vae=vae)

    @torch.no_grad()
    def __call__(
        self,
        sample_fn,
        class_labels: torch.LongTensor,
        latent_shape: tuple,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        latent_mean: Optional[torch.Tensor] = None,
        latent_std: Optional[torch.Tensor] = None,
        latent_multiplier: float = 1.0,
        cfg_scale: float = 1.0,
        cfg_interval_start: float = 0.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ):
        z = torch.randn(latent_shape, generator=generator, device=device, dtype=dtype)
        if cfg_scale > 1.0:
            z = torch.cat([z, z], dim=0)
            y_null = torch.full(
                (class_labels.shape[0],),
                self.transformer.config.num_classes,
                device=device,
                dtype=class_labels.dtype,
            )
            y = torch.cat([class_labels, y_null], dim=0)
        else:
            y = class_labels
        model = self.transformer.module if hasattr(self.transformer, "module") else self.transformer
        model_kwargs = dict(
            y=y,
            cfg_scale=cfg_scale,
            cfg_interval=False,
            cfg_interval_start=cfg_interval_start,
        )
        model_fn = model.forward_with_cfg
        latents = sample_fn(z, model_fn, **model_kwargs)[-1]
        if cfg_scale > 1.0:
            latents, _ = latents.chunk(2, dim=0)
        if latent_mean is not None and latent_std is not None:
            latents = (latents * latent_std) / latent_multiplier + latent_mean
        if self.vae is not None:
            images = self.vae.decode_to_images(latents)
        else:
            images = latents
        if not return_dict:
            return (images,)
        return LightningDiTPipelineOutput(images=images)
