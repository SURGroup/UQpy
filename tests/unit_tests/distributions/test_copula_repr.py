import numpy as np
from UQpy.distributions import copulas


def test_clayton_repr():
    clayton = copulas.Clayton(1.0)
    assert clayton.__repr__() == "Clayton(1.0)"


def test_frank_repr():
    frank = copulas.Frank(2.0)
    assert frank.__repr__() == "Frank(2.0)"


def test_gumbel_repr():
    gumbel = copulas.Gumbel(np.inf)
    assert gumbel.__repr__() == "Gumbel(inf)"
