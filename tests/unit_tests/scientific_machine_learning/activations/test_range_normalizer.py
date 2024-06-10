import pytest
import torch
from UQpy.scientific_machine_learning.activations import RangeNormalizer
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import array_shapes


@given(
    st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    array_shapes(min_dims=1, min_side=2),
)
def test_encode(a, b, size):
    if a == b:
        b += 1
    low = min(a, b)
    high = max(a, b)
    x = torch.normal(0, 5, size=size, dtype=torch.float64)
    normalizer = RangeNormalizer(x, low=low, high=high)
    y = normalizer.encode(x)
    error = 1e-10
    assert torch.all(low - error <= y)
    assert torch.all(y <= high + error)


@given(
    st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    st.floats(min_value=1, max_value=100, allow_nan=False, allow_infinity=False),
    array_shapes(min_dims=1, min_side=2),
)
def test_encode_decode(a, b, size):
    if (b - a) < 0.1:
        b += 0.1
    low = min(a, b)
    high = max(a, b)
    x = torch.normal(0, 5, size=size, dtype=torch.float64)
    normalizer = RangeNormalizer(x, low=low, high=high)
    y = normalizer.encode(x)
    reconstruction = normalizer.decode(y)
    assert torch.allclose(x, reconstruction)


def test_low_greater_than_high():
    x = torch.tensor([1, 2, 3])
    with pytest.raises(ValueError):
        RangeNormalizer(x, 2, 1)


def test_min_equals_max():
    x = torch.ones(100)
    with pytest.raises(RuntimeError):
        RangeNormalizer(x)
