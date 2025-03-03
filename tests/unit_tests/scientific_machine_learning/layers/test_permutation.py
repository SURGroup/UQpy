import torch
import UQpy.scientific_machine_learning as sml
from hypothesis import given
from hypothesis.extra.numpy import array_shapes


@given(array_shapes(min_dims=3, max_dims=3, max_side=8))
def test_forward(size):
    """Test sml.Permutation behaves as torch.permute"""
    dims = (0, 2, 1)
    layer = sml.Permutation(dims)
    x = torch.zeros(size)
    assert layer(x).shape == torch.permute(x, dims).shape


def test_extra_repr():
    layer = sml.Permutation((2, 3, 4))
    assert layer.extra_repr() == "dims=(2, 3, 4)"
