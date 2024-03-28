import torch
import torch.nn
from hypothesis import given
from hypothesis.strategies import integers
from UQpy.scientific_machine_learning.layers.BayesianLayer import BayesianLayer


@given(
    integers(min_value=1, max_value=1_000),
    integers(min_value=1, max_value=1_000),
)
def test_output_shape(in_features, out_features):
    layer = BayesianLayer(in_features, out_features)
    x = torch.ones((in_features,))

    y = layer(x)
    assert y.shape == torch.Size([out_features])


def test_sample_false():
    in_features = 5
    out_features = 10
    layer = BayesianLayer(in_features, out_features)
    layer.train(False)
    x = torch.ones((in_features,))

    y1 = layer(x, sample=False)
    y2 = layer(x, sample=False)
    assert torch.allclose(y1, y2)


def test_sample_true():
    in_features = 5
    out_features = 10
    layer = BayesianLayer(in_features, out_features)
    layer.train(False)
    x = torch.ones((in_features,))

    y1 = layer(x, sample=True)
    y2 = layer(x, sample=True)
    assert not torch.allclose(y1, y2)


def test_train_true():
    in_features = 5
    out_features = 10
    layer = BayesianLayer(in_features, out_features, sample=False)
    layer.train(True)
    x = torch.ones((in_features,))

    y1 = layer(x)
    y2 = layer(x)
    assert not torch.allclose(y1, y2)


def test_extra_repr():
    layer = BayesianLayer(5, 7)
    assert (
        layer.__str__() == "BayesianLayer(in_features=5, out_features=7, sample=True)"
    )
