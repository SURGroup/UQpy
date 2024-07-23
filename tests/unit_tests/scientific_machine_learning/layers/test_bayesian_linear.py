import torch
import torch.nn
from hypothesis import given
from hypothesis.strategies import integers
import UQpy.scientific_machine_learning as sml


@given(
    integers(min_value=1, max_value=1_000),
    integers(min_value=1, max_value=1_000),
)
def test_output_shape(in_features, out_features):
    layer = sml.BayesianLinear(in_features, out_features)
    x = torch.ones((in_features,))

    y = layer(x)
    assert y.shape == torch.Size([out_features])


def test_device():
    """Note if neither cuda nor mps is available, this test will always pass"""
    cpu = torch.device("cpu")
    layer = sml.BayesianLinear(1, 1, device=cpu)
    assert layer.weight_mu.device == cpu
    device = (
        torch.device("cuda", 0)
        if torch.cuda.is_available()
        else torch.device("mps", 0)
        if torch.backends.mps.is_available()
        else "cpu"
    )
    layer.to(device)
    assert layer.weight_mu.device == device


def test_dtype():
    layer = sml.BayesianLinear(1, 1, dtype=torch.float)
    x = torch.rand(1, dtype=torch.float)
    assert layer(x).dtype == torch.float
    layer.to(torch.cfloat)
    x = x.to(torch.cfloat)
    assert layer(x).dtype == torch.cfloat


def test_deterministic_output():
    in_features = 5
    out_features = 10
    layer = sml.BayesianLinear(in_features, out_features)
    layer.sample(False)
    x = torch.ones((in_features,))

    y1 = layer(x)
    y2 = layer(x)
    assert torch.allclose(y1, y2)


def test_probabilistic_output():
    in_features = 5
    out_features = 10
    layer = sml.BayesianLinear(in_features, out_features)
    layer.sample(True)
    x = torch.rand((in_features,))

    y1 = layer(x)
    y2 = layer(x)
    assert not torch.allclose(y1, y2)
