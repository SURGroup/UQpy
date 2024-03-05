import pytest
import numpy as np


@pytest.fixture
def init_test():
    """If you have multiple tests that require the same setup code, consider using a ``pytest.fixture``
    Note this is a typical python function with a return statement, and does *not* contain an ``assert``
    Note this function does *not* start with ``test_``
    """
    a = 1
    b = 2
    return a, b


def test_subtract_equals_one(init_test):
    """Example of a test calling the setup function ``init_test``"""
    a, b = init_test
    assert b - a == 1


def test_subtract_equals_minus_one(init_test):
    """Example of a test calling the setup function ``init_test``"""
    a, b = init_test
    assert a - b == -1


def test_raises_divide_by_zero_error():
    """Example of a standalone test that raises an expected error
    Note that this test *passes* when ZeroDivisionError occurs
    """
    with pytest.raises(ZeroDivisionError):
        result = 2 / 0


@pytest.mark.parametrize("foo,bar", [
    (4, 4),
    (-3.2, -3.2),
    (0, 0),
    (np.array([-1, 12]), np.array([-1, 12]))
])
def test_subtract_equals_zero(foo, bar):
    """Best practice is to use only one ``assert`` statement per test.
    To test multiple different inputs, consider using ``pytest.mark.parameterize`` """
    assert np.allclose(foo - bar, 0)
