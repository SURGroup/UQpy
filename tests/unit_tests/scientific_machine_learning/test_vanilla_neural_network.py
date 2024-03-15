# ToDo: Decide if we should try the hypothesis testing package
# import torch
# import torch.nn as nn
# import hypothesis
# from hypothesis import given
# from hypothesis.strategies import integers
# from hypothesis.extra.numpy import array_shapes
# from UQpy.scientific_machine_learning.neural_networks.VanillaNeuralNetwork import (
#     VanillaNeuralNetwork,
# )
#
#
# @hypothesis.settings(deadline=500)
# @given(
#     integers(min_value=1, max_value=100),
#     integers(min_value=1, max_value=100),
#     integers(min_value=1, max_value=1_000),
# )
# def test_output_shape(in_dim, out_dim, n_samples):
#     simple_network = nn.Linear(in_dim, out_dim)
#     model = VanillaNeuralNetwork(simple_network)
#     x = torch.ones((n_samples, in_dim))
#
#     prediction = model(x)
#     assert prediction.shape == torch.Size([n_samples, out_dim])
#
#
# @hypothesis.settings(deadline=500)
# @given(array_shapes(min_dims=1, max_dims=4), integers(min_value=1, max_value=100))
# def test_no_bias(shape, out_dim):
#     """With no bias, f(0) = 0 for any array shape"""
#     x = torch.zeros(shape)
#     simple_network = nn.Linear(x.shape[-1], out_dim, bias=False)
#     model = VanillaNeuralNetwork(simple_network)
#
#     prediction = model(x)
#     assert torch.allclose(prediction, torch.zeros_like(prediction))
