# ToDo: write this test with the new generalized JS in functional.
# ToDo: write device test
# import pytest
# import torch
# import torch.nn as nn
# import UQpy.scientific_machine_learning as sml
# from hypothesis import given, strategies as st
#
#
# @given(width=st.integers(min_value=1, max_value=10))
# def test_reduction_shape(width):
#     network = nn.Sequential(
#         sml.BayesianLinear(1, width),
#         nn.ReLU(),
#         sml.BayesianLinear(width, width),
#         nn.ReLU(),
#         sml.BayesianLinear(width, 1),
#     )
#     model = sml.FeedForwardNeuralNetwork(network)
#     divergence_loss = sml.GeneralizedJensenShannonDivergence()
#     divergence = divergence_loss(model)
#     assert divergence.shape == torch.Size()
#
#
# def test_reduction_none_raises_error():
#     with pytest.raises(ValueError):
#         sml.GeneralizedJensenShannonDivergence(reduction="sum")
