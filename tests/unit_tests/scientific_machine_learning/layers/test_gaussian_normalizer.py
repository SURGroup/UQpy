import pytest
import torch
import UQpy.scientific_machine_learning as sml
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import array_shapes


@given(
    mean=st.floats(
        min_value=-100, max_value=100, allow_nan=False, allow_infinity=False
    ),
    std=st.floats(min_value=0.01, max_value=100, allow_nan=False, allow_infinity=False),
    size=array_shapes(min_dims=1, min_side=10),
)
def test_encode(mean, std, size):
    x = torch.normal(mean, std, size=size, dtype=torch.float64)
    normalizer = sml.GaussianNormalizer(x)
    y = normalizer(x)
    assert torch.isclose(torch.mean(y), torch.tensor([0.0], dtype=torch.float64))
    assert torch.isclose(torch.std(y), torch.tensor([1.0], dtype=torch.float64))


@given(
    mean=st.floats(
        min_value=-100, max_value=100, allow_nan=False, allow_infinity=False
    ),
    std=st.floats(min_value=0.01, max_value=100, allow_nan=False, allow_infinity=False),
    size=array_shapes(min_dims=1, min_side=10),
)
def test_encode_decode(mean, std, size):
    x = torch.normal(mean, std, size=size, dtype=torch.float64)
    normalizer = sml.GaussianNormalizer(x)
    y = normalizer(x)
    normalizer.decode()
    x_reconstruction = normalizer(y)
    assert torch.allclose(x, x_reconstruction)


def test_nan_raises_error():
    x = torch.tensor([1.0, torch.nan, 3.0])
    with pytest.raises(RuntimeError):
        sml.GaussianNormalizer(x)


def test_inf_raises_error():
    x = torch.tensor([1.0, torch.inf, 3.0])
    with pytest.raises(RuntimeError):
        sml.GaussianNormalizer(x)
