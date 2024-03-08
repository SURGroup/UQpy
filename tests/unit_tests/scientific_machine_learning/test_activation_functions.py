import torch
from UQpy.scientific_machine_learning.activation_functions.ProbabilisticDropout import ProbabilisticDropout


def test_dropout_p1():
    """Probability ``p=1.0`` should set *all* tensor elements to zero"""
    dropout = ProbabilisticDropout(p=1.0, active=True)
    t = torch.ones((3, 3))
    assert (dropout(t) == torch.zeros_like(t)).all()


def test_dropout_p0():
    """Probability ``p=0.0`` should set *no* tensor elements to zero"""
    dropout = ProbabilisticDropout(p=0.0, active=True)
    t = torch.ones((3, 3))
    assert (dropout(t) == t).all()


def test_dropout_active_false():
    """Setting ``active=False`` should affect *no* tensor elements"""
    dropout = ProbabilisticDropout(p=1.0, active=False)
    t = torch.ones((3, 3))
    assert (dropout(t) == t).all()


def test_dropout_str():
    """Confirm the ``extra_repr`` method inherited from ``torch.nn.Module`` is correctly implemented"""
    dropout = ProbabilisticDropout(p=0.5, active=False)
    assert dropout.__str__() == 'ProbabilisticDropout(p=0.5, active=False)'
