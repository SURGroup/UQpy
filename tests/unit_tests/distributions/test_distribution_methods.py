from UQpy.distributions import *
import numpy as np

# Test all functions for one type of continuous distribution: uniform
dist_continuous = Uniform(loc=1.0, scale=2.0)


def test_get_params():
    np.testing.assert_allclose(dist_continuous.get_parameters()["loc"], 1.0)


def test_update_params():
    dist = Uniform(loc=1.0, scale=2.0)
    dist.update_parameters(loc=2.0)
    np.testing.assert_allclose(dist.get_parameters()["loc"], 2.0)


def test_continuous_pdf():
    np.testing.assert_allclose(dist_continuous.pdf(x=1.5), 0.5)


def test_continuous_cdf():
    np.testing.assert_allclose(dist_continuous.cdf(x=1.5), 0.25)


def test_continuous_log_pdf():
    np.testing.assert_allclose(dist_continuous.log_pdf(x=1.5), -0.693, atol=1e-3)


def test_continuous_icdf():
    np.testing.assert_allclose(dist_continuous.icdf(x=0.9), 2.8)


def test_continuous_rvs():
    samples = dist_continuous.rvs(nsamples=2, random_state=123)
    np.testing.assert_allclose(samples, np.array([[2.393], [1.572]]), atol=1e-3)


def test_continuous_fit():
    dict_fit = Uniform(loc=None, scale=None).fit(data=[1.5, 2.5, 3.5])
    assert isinstance(dict_fit, dict)
    np.testing.assert_allclose(dict_fit["loc"], 1.5)
    np.testing.assert_allclose(dict_fit["scale"], 2.0)


def test_continuous_moments():
    np.testing.assert_allclose(dist_continuous.moments(moments2return="m"), 2.0)


# Test all functions for one type of discrete distribution: binomial
dist_discrete = Binomial(n=5, p=0.2)


def test_discrete_pmf():
    np.testing.assert_allclose(dist_discrete.pmf(x=2.0), 0.205, atol=1e-3)


def test_discrete_cdf():
    np.testing.assert_allclose(dist_discrete.cdf(x=2.0), 0.942, atol=1e-3)


def test_discrete_log_pmf():
    np.testing.assert_allclose(dist_discrete.log_pmf(x=2.0), -1.586, atol=1e-3)


def test_discrete_icdf():
    np.testing.assert_allclose(dist_discrete.icdf(0.9), 2.0)


def test_discrete_rvs():
    samples = dist_discrete.rvs(nsamples=2, random_state=123)
    np.testing.assert_allclose(samples, np.array([[1.0], [0.0]]), atol=1e-3)


def test_discrete_moments():
    np.testing.assert_allclose(dist_discrete.moments(moments2return="m"), 1.0)


# Test functions for Copula


def test_update_params_copula():
    copula = Gumbel(theta=2.0)
    copula.update_parameters(theta=1.0)
    np.testing.assert_allclose(copula.get_parameters()["theta"], 1.0)
