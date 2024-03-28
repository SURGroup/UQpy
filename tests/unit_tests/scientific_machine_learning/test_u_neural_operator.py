import torch
import torch.nn as nn
from UQpy.scientific_machine_learning.neural_networks.UNeuralNetwork import UNeuralNetwork


def test_output_shape():
    n_filters = [8, 3, 1, 9, 5]
    kernel_size = 3
    out_channels = 12
    model = UNeuralNetwork(n_filters, kernel_size, out_channels)
    n, c, h, w = 1, 8, 128, 128  # number of data, channels, height, width
    x = torch.ones((n, c, h, w))
    y = model(x)
    print(x.shape, y.shape)
    # ToDo: expected shape of y is (n, out_channels, h, w)