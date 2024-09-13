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


def test_dim_int():
    """Consider a 2d tensor of shape (3, 100). When dim=1, each row should be normalized to mean=0, std=1."""
    dim = 1
    x = 10 * torch.randn((3, 100)) + 50
    normalizer = sml.GaussianNormalizer(x, dim=dim)
    y = normalizer(x)
    assert torch.allclose(torch.mean(y, dim=dim), torch.tensor([0.0]), atol=1e-6)
    assert torch.allclose(torch.std(y, dim=dim), torch.tensor([1.0]))


def test_dim_tuple():
    """Consider a tensor of shape (3, 4, 100, 200).
    When dim=(2, 3), the dimensions of shape 100 and 200 reduced,
    resulting in each index [row, col, :, :] normalized to mean=0, std=1
    """
    dim = (2, 3)
    x = 10 * torch.randn((3, 4, 100, 200), dtype=torch.float64) + 50  # test data far outside [0, 1]
    normalizer = sml.GaussianNormalizer(x, dim=dim)
    y = normalizer(x)
    assert torch.allclose(torch.mean(y, dim=dim), torch.tensor([0.0], dtype=torch.float64))
    assert torch.allclose(torch.std(y, dim=dim), torch.tensor([1.0], dtype=torch.float64))


def test_nan_raises_error():
    x = torch.tensor([1.0, torch.nan, 3.0])
    with pytest.raises(RuntimeError):
        sml.GaussianNormalizer(x)


def test_inf_raises_error():
    x = torch.tensor([1.0, torch.inf, 3.0])
    with pytest.raises(RuntimeError):
        sml.GaussianNormalizer(x)
