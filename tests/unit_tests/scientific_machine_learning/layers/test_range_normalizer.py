import pytest
import torch
import UQpy.scientific_machine_learning as sml
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import array_shapes

settings.register_profile("fast", max_examples=1)
settings.load_profile("fast")


@given(
    shift=st.floats(
        min_value=-100, max_value=100, allow_nan=False, allow_infinity=False
    ),
    width=st.floats(
        min_value=0.01, max_value=100, allow_nan=False, allow_infinity=False
    ),
    size=array_shapes(min_dims=1, min_side=10),
)
def test_encode(shift, width, size):
    x = torch.rand(size=size, dtype=torch.float64)
    x = (width * x) + shift
    normalizer = sml.RangeNormalizer(x)
    y = normalizer(x)
    assert torch.isclose(y.min(), torch.tensor([0.0], dtype=torch.float64))
    assert torch.isclose(y.max(), torch.tensor([1.0], dtype=torch.float64))


@given(
    shift=st.floats(
        min_value=-100, max_value=100, allow_nan=False, allow_infinity=False
    ),
    width=st.floats(
        min_value=0.01, max_value=100, allow_nan=False, allow_infinity=False
    ),
    size=array_shapes(min_dims=1, min_side=10),
)
def test_encode_decode(shift, width, size):
    x = torch.rand(size=size, dtype=torch.float64)
    x = (width * x) + shift
    normalizer = sml.RangeNormalizer(x)
    normalizer.encode()
    y = normalizer(x)
    normalizer.decode()
    x_reconstruction = normalizer(y)
    assert torch.allclose(x, x_reconstruction)


@pytest.mark.parametrize("row", (0, 1, 2))
def test_dim_int(row):
    """Consider a 2d tensor of shape (3, 100). When dim=1, each row should be normalized to [0, 1]."""
    x = torch.randn((3, 100)) + 10  # test data far outside [0, 1]
    normalizer = sml.RangeNormalizer(x, dim=1)
    y = normalizer(x)
    assert torch.isclose(y[row, :].min(), torch.tensor(0.0))
    assert torch.isclose(y[row, :].max(), torch.tensor(1.0))


@pytest.mark.parametrize("row", (0, 1, 2))
@pytest.mark.parametrize("col", (0, 1, 2, 3))
def test_dim_tuple(row, col):
    """Consider a tensor of shape (3, 4, 100, 200).
    When dim=(2, 3), the dimensions of shape 100 and 200 reduced,
    resulting in each index [row, col, :, :] normalized to [0, 1]
    """
    x = torch.randn((3, 4, 100, 200)) + 10  # test data far outside [0, 1]
    normalizer = sml.RangeNormalizer(x, dim=(2, 3))
    y = normalizer(x)
    assert torch.isclose(y[row, col].min(), torch.tensor(0.0))
    assert torch.isclose(y[row, col].max(), torch.tensor(1.0))


def test_nan_raises_error():
    with pytest.raises(RuntimeError):
        x = torch.tensor([1, torch.nan, 3])
        sml.RangeNormalizer(x)


def test_inf_raises_error():
    with pytest.raises(RuntimeError):
        x = torch.tensor([1, 2, float("inf")])
        sml.RangeNormalizer(x)


def test_low_greater_than_high_raises_error():
    x = torch.tensor([1, 2, 3])
    with pytest.raises(ValueError):
        sml.RangeNormalizer(x, low=2, high=1)


def test_min_equals_max_raises_error():
    x = torch.ones(100)
    with pytest.raises(RuntimeError):
        sml.RangeNormalizer(x)


def test_extra_repr():
    """Customize all input options to test the extra_repr method correctly displays non-default inputs"""
    x = torch.tensor([[1, 2, 3]])
    normalizer = sml.RangeNormalizer(x, encoding=False, low=1, high=2, dim=1)
    assert normalizer.extra_repr() == "encoding=False, low=1, high=2, dim=1"