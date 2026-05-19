import pytest

torch = pytest.importorskip("torch")

import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from diffusers.models.transformers import LightningDiTTransformer2DModel
from diffusers.schedulers import LightningDiTFlowMatchScheduler


def test_lightningdit_transformer_forward():
    model = LightningDiTTransformer2DModel(
        input_size=4,
        patch_size=1,
        in_channels=4,
        hidden_size=32,
        depth=2,
        num_heads=4,
        num_classes=10,
        use_rope=False,
    )
    latents = torch.randn(2, 4, 4, 4)
    timesteps = torch.tensor([0.0, 0.5])
    class_labels = torch.tensor([1, 2])

    output = model(latents, timesteps, class_labels)

    assert output.sample.shape == latents.shape


def test_scheduler_ode_step():
    scheduler = LightningDiTFlowMatchScheduler()
    sample = torch.ones(1, 4, 2, 2)
    velocity = torch.full_like(sample, 2.0)

    output = scheduler.step(velocity, torch.tensor([0.0]), sample, torch.tensor([0.25]))

    assert torch.allclose(output.prev_sample, torch.full_like(sample, 1.5))


def test_cfg_applies_to_first_three_channels_only():
    batch = 2
    channels = 5
    model_output = torch.arange(batch * channels, dtype=torch.float32).view(batch, channels, 1, 1)
    model_output = torch.cat([model_output, model_output], dim=0)
    guided = LightningDiTTransformer2DModel.apply_classifier_free_guidance(model_output, guidance_scale=2.0, cfg_channels=3)
    assert guided.shape[0] == batch
    assert guided.shape[1] == channels
