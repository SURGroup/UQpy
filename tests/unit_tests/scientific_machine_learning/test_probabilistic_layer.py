import torch
import torch.nn
from hypothesis import given
from hypothesis.strategies import integers
from UQpy.distributions.collection import Gamma
from UQpy.scientific_machine_learning.layers.ProbabilisticLayer import (
    ProbabilisticLayer,
)


@given(integers(min_value=1, max_value=10), integers(min_value=1, max_value=10))
def test_output_shape(in_features, out_features):
    layer = ProbabilisticLayer(in_features, out_features)
    x = torch.ones((in_features,))

    y = layer(x)
    assert y.shape == torch.Size([out_features])


@given(integers(min_value=1, max_value=10), integers(min_value=1, max_value=10))
def test_weight_normal_parameters(in_features, out_features):
    layer = ProbabilisticLayer(in_features, out_features, bias_distribution=None)

    assert layer.weight_parameters.shape == torch.Size((in_features * out_features, 2))
    assert (
        layer.weight_parameters[:, 0] == layer.weight_distribution.parameters["loc"]
    ).all()
    assert (
        layer.weight_parameters[:, 1] == layer.weight_distribution.parameters["scale"]
    ).all()


@given(integers(min_value=1, max_value=10), integers(min_value=1, max_value=10))
def test_bias_normal_parameters(in_features, out_features):
    layer = ProbabilisticLayer(
        in_features,
        out_features,
    )
    assert layer.bias_parameters.shape == torch.Size((out_features, 2))
    assert (
        layer.bias_parameters[:, 0] == layer.bias_distribution.parameters["loc"]
    ).all()
    assert (
        layer.bias_parameters[:, 1] == layer.bias_distribution.parameters["scale"]
    ).all()


@given(integers(min_value=1, max_value=10), integers(min_value=1, max_value=10))
def test_weight_gamma_parameters(in_features, out_features):
    layer = ProbabilisticLayer(
        in_features,
        out_features,
        weight_distribution=Gamma(a=1, loc=2, scale=3),
        bias_distribution=None,
    )
    assert layer.weight_parameters.shape == torch.Size((in_features * out_features, 3))
    assert (
        layer.weight_parameters[:, 0] == layer.weight_distribution.parameters["a"]
    ).all()
    assert (
        layer.weight_parameters[:, 1] == layer.weight_distribution.parameters["loc"]
    ).all()
    assert (
        layer.weight_parameters[:, 2] == layer.weight_distribution.parameters["scale"]
    ).all()


@given(integers(min_value=1, max_value=10), integers(min_value=1, max_value=10))
def test_bias_gamma_parameters(in_features, out_features):
    layer = ProbabilisticLayer(
        in_features,
        out_features,
        bias_distribution=Gamma(a=4, loc=5, scale=6),
    )
    assert layer.bias_parameters.shape == torch.Size((out_features, 3))
    assert (
        layer.bias_parameters[:, 0] == layer.bias_distribution.parameters["a"]
    ).all()
    assert (
        layer.bias_parameters[:, 1] == layer.bias_distribution.parameters["loc"]
    ).all()
    assert (
        layer.bias_parameters[:, 2] == layer.bias_distribution.parameters["scale"]
    ).all()


def test_output_is_deterministic():
    layer = ProbabilisticLayer(5, 7)
    layer.train(False)
    layer.sample = False

    x = torch.ones(5)
    y1 = layer(x)
    y2 = layer(x)
    assert (y1 == y2).all()


def test_output_is_random_via_sample():
    layer = ProbabilisticLayer(5, 7)
    layer.train(False)
    layer.sample = True

    x = torch.ones(5)
    y1 = layer(x)
    y2 = layer(x)
    assert (y1 != y2).all()


def test_output_is_random_via_train():
    layer = ProbabilisticLayer(5, 7)
    layer.train(True)
    layer.sample = False

    x = torch.ones(5)
    y1 = layer(x)
    y2 = layer(x)
    assert (y1 != y2).all()


def test_bias_none():
    """With zero bias f(0) = 0, and it is very likely f(1) != 0"""
    in_features = 10
    out_features = 20
    layer = ProbabilisticLayer(in_features, out_features, bias_distribution=None)
    x0 = torch.zeros(in_features)
    x1 = torch.ones(in_features)
    assert torch.allclose(layer(x0), torch.zeros(out_features))
    assert torch.not_equal(layer(x1), torch.zeros(out_features)).all()


def test_sample_weight():
    layer = ProbabilisticLayer(5, 7)
    old_weight = torch.clone(layer.weight)
    layer._sample_weight()
    assert torch.not_equal(old_weight, layer.weight).all()


def test_sample_bias():
    layer = ProbabilisticLayer(5, 7)
    old_bias = torch.clone(layer.bias)
    layer._sample_bias()
    assert torch.not_equal(old_bias, layer.bias).all()
