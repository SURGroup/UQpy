from UQpy.distributions import Uniform
from UQpy.surrogates import *
import numpy as np

np.random.seed(1)
dist = Uniform(location=0, scale=10)
max_degree, n_samples = 2, 10
polys = Polynomials(distributions=dist, degree=max_degree)


def func(x):
    return x * np.sin(x) / 10

x = dist.rvs(n_samples)
y = func(x)

def poly_func(x):
    p = polys.evaluate(x)
    return p

def pce_model(polys, x, y):
    lstsq = LeastSquareRegression(polynomials=polys)
    pce = pce(method=lstsq)
    pce.fit(x, y)
    return pce

def pce_predict(x, c):
    x_ = poly_func(x)
    return x_.dot(c)


# Unit tests

def test_1():
    """
    Test Polynomials class
    """
    assert round(poly_func(x)[1, 1], 3) == 0.763


def test_2():
    """
    Test polynomial_chaos class coefficients
    """
    assert round(float(pce_model(polys, x, y).C[1]), 3) == 0.328


def test_3():
    """
    Test polynomial_chaos class prediction
    """
    model = pce_model(polys, x, y)
    assert round(float(pce_predict(x, model.C)[1]), 3) == 0.338


def test_4():
    """
    Test Error Estimation class
    """
    x_val = x + np.random.rand(1)
    y_val = func(x_val)
    error = ErrorEstimation(pce_surrogate=pce_model(polys, x, y))
    assert round(error.validation(x_val, y_val), 3) == 0.432
