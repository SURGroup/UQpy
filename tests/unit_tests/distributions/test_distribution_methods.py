import numpy as np
from UQpy.distributions import *


# Test all functions for one type of continuous distribution: uniform
dist_continuous = Uniform(loc=1.0, scale=2.0)


def test_get_params():
    assert dist_continuous.get_parameters()["loc"] == 1.0


def test_update_params():
    dist = Uniform(loc=1.0, scale=2.0)
    dist.update_parameters(loc=2.0)
    assert dist.get_parameters()["loc"] == 2.0


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
    assert dict_fit == {"loc": 1.5, "scale": 2.0}


def test_continuous_moments():
    assert dist_continuous.moments(moments2return="m") == 2.0


# Test all functions for one type of discrete distribution: binomial
dist_discrete = Binomial(n=5, p=0.2)


def test_discrete_pmf():
    assert np.round(dist_discrete.pmf(x=2.0), 3) == 0.205


def test_discrete_cdf():
    assert np.round(dist_discrete.cdf(x=2.0), 3) == 0.942


def test_discrete_log_pmf():
    assert np.round(dist_discrete.log_pmf(x=2.0), 3) == -1.586


def test_discrete_icdf():
    assert dist_discrete.icdf(0.9) == 2.0


def test_discrete_rvs():
    samples = dist_discrete.rvs(nsamples=2, random_state=123)
    assert np.all(np.round(samples, 3) == np.array([1.0, 0.0]).reshape((2, 1)))


def test_discrete_moments():
    assert dist_discrete.moments(moments2return="m") == 1.0


# Test functions for Copula
def test_update_params_copula():
    copula = Gumbel(theta=2.0)
    copula.update_parameters(theta=1.0)
    assert copula.get_parameters()["theta"] == 1.0
