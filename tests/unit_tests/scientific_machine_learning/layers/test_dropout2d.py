import torch
import UQpy.scientific_machine_learning as sml
from hypothesis import given
from hypothesis.extra.numpy import array_shapes

func = sml.ProbabilisticDropout2d
shapes = array_shapes(min_dims=3, max_dims=4, min_side=1, max_side=32)


@given(shapes)
def test_set_drop_false(shape):
    """Test no elements are set to zero and dropout is identity function"""
    dropout = func(dropping=True)
    dropout.drop(False)
    x = torch.ones(shape)
    assert torch.all(x == dropout(x))


@given(shapes)
def test_dropping_false(shape):
    """Test no elements are set to zero and dropout is identity function"""
    dropout = func(dropping=False)
    x = torch.ones(shape)
    assert torch.all(x == dropout(x))


def test_dropping_true():
    """Test output is probabilistic on a large array"""
    dropout = func(dropping=True)
    x = torch.ones(100, 100)
    y1 = dropout(x)
    y2 = dropout(x)
    assert not torch.allclose(y1, y2)


@given(shapes)
def test_p_zero(shape):
    """Test no elements are dropped when drop_rate is zero"""
    dropout = func(p=0.0, dropping=True)
    x = torch.ones(shape)
    assert torch.all(x == dropout(x))


@given(shapes)
def test_p_one(shape):
    """Test all elements are dropped when drop_rate is one"""
    dropout = func(p=1.0, dropping=True)
    x = torch.ones(shape)
    assert torch.all(torch.zeros_like(x) == dropout(x))
