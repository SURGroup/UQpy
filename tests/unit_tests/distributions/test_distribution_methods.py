from UQpy.distributions import *
import numpy as np
import pytest
import scipy


# Test all functions for one type of continuous distribution: uniform
dist_continuous = Uniform(loc=1., scale=2.)


def test_get_params():
    assert dist_continuous.get_parameters()['loc'] == 1.


def test_update_params():
    dist = Uniform(loc=1., scale=2.)
    dist.update_parameters(loc=2.)
    assert dist.get_parameters()['loc'] == 2.


def test_continuous_pdf():
    assert dist_continuous.pdf(x=1.5) == 0.5


def test_continuous_cdf():
    assert dist_continuous.cdf(x=1.5) == 0.25


def test_continuous_log_pdf():
    assert np.round(dist_continuous.log_pdf(x=1.5), 3) == -0.693


def test_continuous_icdf():
    assert dist_continuous.icdf(x=0.9) == 2.8


def test_continuous_rvs():
    samples = dist_continuous.rvs(nsamples=2, random_state=123)
    assert np.all(np.round(samples, 3) == np.array([2.393, 1.572]).reshape((2, 1)))


def test_continuous_fit():
    dict_fit = Uniform(loc=None, scale=None).fit(data=[1.5, 2.5, 3.5])
    assert dict_fit == {'loc': 1.5, 'scale': 2.0}


def test_continuous_moments():
    assert dist_continuous.moments(moments2return='m') == 2.


# Test all functions for one type of discrete distribution: binomial
dist_discrete = Binomial(n=5, p=0.2)


def test_discrete_pmf():
    assert np.round(dist_discrete.pmf(x=2.), 3) == 0.205


def test_discrete_cdf():
    assert np.round(dist_discrete.cdf(x=2.), 3) == 0.942


def test_discrete_log_pmf():
    assert np.round(dist_discrete.log_pmf(x=2.), 3) == -1.586


def test_discrete_icdf():
    assert dist_discrete.icdf(0.9) == 2.


def test_discrete_rvs():
    samples = dist_discrete.rvs(nsamples=2, random_state=123)
    assert np.all(np.round(samples, 3) == np.array([1., 0.]).reshape((2, 1)))


def test_discrete_moments():
    assert dist_discrete.moments(moments2return='m') == 1.


# Test functions for Copula


def test_update_params_copula():
    copula = Gumbel(theta=2.)
    copula.update_parameters(theta=1.)
    assert copula.get_parameters()['theta'] == 1.


# Test function for custom implementation of Uniform PDF and CDF
@pytest.mark.parametrize("value,expected_probability", [
    (-4, 0),
    (0, 1),
    (0.5, 1),
    (1, 1),
    (12, 0),
    ([-1, 0, 0.5, 1, 2], [0, 1, 1, 1, 0]),
    ((-1, 0, 0.5, 1, 2), (0, 1, 1, 1, 0)),
    (np.array([-1, 0, 0.5, 1, 2]), np.array([0, 1, 1, 1, 0]))
])
def test_uniform_pdf(value, expected_probability):
    uniform = Uniform()
    assert (np.isclose(uniform.pdf(value), expected_probability, equal_nan=True)).all()
    # assert all(uniform.pdf(value) == expected_probability)


@pytest.mark.parametrize("value,expected_probability", [
    (-0.1245, 0),
    (0.4, 0.4),
    (3.14159, 1),
    ([-4, 0.2, 0.75, 1, 6], [0, 0.2, 0.75, 1, 1]),
    ((-4, 0.2, 0.75, 1, 6), (0, 0.2, 0.75, 1, 1)),
    (np.array([-4, 0.2, 0.75, 1, 6]), np.array([0, 0.2, 0.75, 1, 1])),
])
def test_uniform_cdf(value, expected_probability):
    uniform = Uniform()
    assert (np.isclose(uniform.cdf(value), expected_probability, equal_nan=True)).all()
    # assert all(uniform.cdf(value) == expected_probability)


@pytest.mark.parametrize("probability,expected_value", [
    (-1, np.nan),
    (0, 0),
    (0.123, 0.123),
    (1, 1),
    (1.2, np.nan),
    ((-0.5, 0, 0.5, 1, 1.5), (np.nan, 0, 0.5, 1, np.nan)),
    ([-0.5, 0, 0.5, 1, 1.5], [np.nan, 0, 0.5, 1, np.nan]),
    (np.array([-0.5, 0, 0.5, 1, 1.5]), np.array([np.nan, 0, 0.5, 1, np.nan]))
])
def test_uniform_icdf(probability, expected_value):
    uniform = Uniform()
    assert (np.isclose(uniform.icdf(probability), expected_value, equal_nan=True)).all()


@pytest.mark.parametrize("value", [-3, -2.5, -1, 0, 1, 2.415, 6.168, np.array([-8, -4.13, -1, 0, 2])])
def test_normal_pdf(value):
    normal = Normal()
    assert (np.isclose(normal.pdf(value), scipy.stats.norm.pdf(value))).all()


@pytest.mark.parametrize("value", [-4, -2.7, -0.4, 0, 0.4321, 2, 8, np.array([-3, -2, 0, 1, 4])])
def test_normal_cdf(value):
    normal = Normal()
    assert (np.isclose(normal.cdf(value), scipy.stats.norm.cdf(value))).all()
