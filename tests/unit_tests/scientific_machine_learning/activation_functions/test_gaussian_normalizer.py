import torch
from UQpy.scientific_machine_learning.activation_functions import GaussianNormalizer
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import array_shapes


@given(
    st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    array_shapes(min_dims=1, min_side=2),
)
def test_encode(scale, shift, size):
    x = torch.rand(size, dtype=torch.float64)
    x = (x * scale) + shift
    normalizer = GaussianNormalizer(x)
    y = normalizer.encode(x)
    assert torch.isclose(torch.mean(y), torch.tensor([0], dtype=torch.float64))
    assert torch.isclose(torch.std(y), torch.tensor([1], dtype=torch.float64))


@given(array_shapes(min_dims=1, min_side=2))
def test_encode_decode(size):
    x = torch.rand(size, dtype=torch.float64)
    normalizer = GaussianNormalizer(x)
    y = normalizer.encode(x)
    reconstruction = normalizer.decode(y)
    assert torch.allclose(x, reconstruction)
