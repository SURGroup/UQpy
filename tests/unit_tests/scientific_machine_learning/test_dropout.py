import torch
from UQpy.scientific_machine_learning.layers import Dropout
from hypothesis import given
from hypothesis.extra.numpy import array_shapes


@given(array_shapes(min_dims=1, max_dims=4, min_side=1, max_side=100))
def test_set_drop_false(shape):
    """Test no elements are set to zero"""
    dropout = Dropout(dropping=True)
    dropout.drop(False)
    x = torch.ones(shape)
    assert torch.all(x == dropout(x))


@given(array_shapes(min_dims=1, max_dims=4, min_side=1, max_side=100))
def test_dropping_false(shape):
    """Test no elements are set to zero and dropout behaves like identity"""
    dropout = Dropout(dropping=False)
    x = torch.one(shape)
    assert torch.all(x == dropout(x))


def test_dropping_true():
    """Test output is probabilistic on a large array"""
    dropout = Dropout(dropping=True)
    x = torch.ones(100, 100)
    y1 = dropout(x)
    y2 = dropout(x)
    assert not torch.allclose(y1, y2)


@given(array_shapes(min_dims=1, max_dims=4, min_side=1, max_side=10))
def test_drop_rate_one(shape):
    """Test all elements are dropped when drop_rate is one"""
    dropout = Dropout(drop_rate=1.0, dropping=True)
    x = torch.ones(shape)
    assert torch.all(torch.zeros_like(x) == dropout(x))


@given(array_shapes(min_dims=1, max_dims=4, min_side=1, max_side=10))
def test_drop_rate_zero(shape):
    """Test no elements are dropped when drop_rate is zero"""
    dropout = Dropout(drop_rate=0.0, dropping=True)
    x = torch.ones(shape)
    assert torch.all(x == dropout(x))
