import pytest

torch = pytest.importorskip("torch")

from lightningdit_diffusers import LightningDiT_models, create_transport, Sampler


def test_lightningdit_forward():
    model = LightningDiT_models["LightningDiT-B/1"](
        input_size=16,
        num_classes=10,
        in_channels=4,
        use_qknorm=False,
        use_swiglu=False,
        use_rope=False,
        use_rmsnorm=False,
    )
    x = torch.randn(2, 4, 16, 16)
    t = torch.rand(2)
    y = torch.randint(0, 10, (2,))
    out = model(x, t, y)
    assert out.shape == x.shape


def test_create_transport_sampler():
    tr = create_transport("Linear", "velocity", None, None, None)
    sampler = Sampler(tr)
    fn = sampler.sample_ode(sampling_method="euler", num_steps=5, atol=1e-6, rtol=1e-3)
    assert callable(fn)
