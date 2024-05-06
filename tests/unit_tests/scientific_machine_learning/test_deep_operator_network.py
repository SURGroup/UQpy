import torch
import torch.nn as nn
import hypothesis
from hypothesis import given
from hypothesis.strategies import integers
from UQpy.scientific_machine_learning.neural_networks.DeepOperatorNetwork import (
    DeepOperatorNetwork,
)


@hypothesis.settings(deadline=500)
@given(integers(min_value=10, max_value=1_000), integers(min_value=1, max_value=128))
def test_1d_output_shape(x_resolution, n_samples):
    x_dimension = 1
    x = torch.ones((n_samples, x_resolution, x_dimension))
    u_x = torch.ones((n_samples, x_resolution))

    model = DeepOperatorNetwork(nn.Linear(x_resolution, 7), nn.Linear(x_dimension, 7), 1)

    prediction = model(x, u_x)
    assert prediction.shape == torch.Size([n_samples, x_resolution])


@hypothesis.settings(deadline=500)
@given(integers(min_value=10, max_value=1_000), integers(min_value=1, max_value=128))
def test_2d_output_shape(x_resolution, n_samples):
    x_dimension = 2
    x = torch.ones((n_samples, x_resolution, x_dimension))
    u_x = torch.ones((n_samples, x_resolution))

    model = DeepOperatorNetwork(nn.Linear(x_resolution, 10), nn.Linear(x_dimension, 10), 1)

    prediction = model(x, u_x)
    assert prediction.shape == torch.Size([n_samples, x_resolution])


@hypothesis.settings(deadline=500)
@given(integers(min_value=10, max_value=1_000), integers(min_value=1, max_value=128))
def test_3d_output_shape(x_resolution, n_samples):
    x_dimension = 3
    x = torch.ones((n_samples, x_resolution, x_dimension))
    u_x = torch.ones((n_samples, x_resolution))

    model = DeepOperatorNetwork(nn.Linear(x_resolution, 10), nn.Linear(x_dimension, 10), 1)

    prediction = model(x, u_x)
    assert prediction.shape == torch.Size([n_samples, x_resolution])


def test_no_bias():
    """With no bias, f(0) = 0"""
    x_resolution = 100
    x = torch.linspace(0, 1, x_resolution).reshape(1, x_resolution, 1)
    u_x = torch.zeros((1, x_resolution))

    model = DeepOperatorNetwork(nn.Linear(x_resolution, 7, bias=False), nn.Linear(1, 7, bias=False), 1)

    prediction = model(x, u_x)
    assert torch.allclose(prediction, torch.zeros_like(prediction))


@hypothesis.settings(deadline=500)
@given(integers(min_value=10, max_value=1_000), integers(min_value=1, max_value=128))
def test_2outputs_2d_output_shape(x_resolution, n_samples):
    x_dimension = 2
    x = torch.ones((n_samples, x_resolution, x_dimension))
    u_x = torch.ones((n_samples, x_resolution))

    model = DeepOperatorNetwork(nn.Linear(x_resolution, 10), nn.Linear(x_dimension, 10), 2)

    prediction1, prediction2 = model(x, u_x)
    assert prediction1.shape == torch.Size([n_samples, x_resolution])
    assert prediction2.shape == torch.Size([n_samples, x_resolution])
