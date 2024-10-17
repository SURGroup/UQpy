# import pytest
import numpy as np
from scipy import stats
from UQpy.distributions import Uniform
import hypothesis.strategies as st
from hypothesis import given
from hypothesis.extra.numpy import array_shapes

uniform = Uniform()
scipy_uniform = stats.uniform()


@given(st.floats(allow_nan=True))
def test_uniform_pdf_float(x):
    """Test custom implementation of uniform pdf on float inputs"""
    pdf = uniform.pdf(x)
    scipy_pdf = scipy_uniform.pdf(x)
    assert isinstance(pdf, float)
    assert np.allclose(pdf, scipy_pdf, equal_nan=True)


@given(array_shapes(min_dims=1, min_side=1))
def test_uniform_pdf_array(size):
    x = np.random.normal(0, 1, size=size)
    pdf = uniform.pdf(x)
    scipy_pdf = scipy_uniform.pdf(x)
    assert isinstance(pdf, np.ndarray)
    assert np.allclose(pdf, scipy_pdf, equal_nan=True)


@given(st.floats(allow_nan=True))
def test_uniform_cdf_float(x):
    pdf = uniform.pdf(x)
    scipy_pdf = scipy_uniform.pdf(x)
    assert isinstance(pdf, float)
    assert np.allclose(pdf, scipy_pdf, equal_nan=True)


@given(array_shapes(min_dims=1, min_side=1))
def test_uniform_cdf_array(size):
    x = np.random.normal(0, 1, size=size)
    pdf = uniform.pdf(x)
    scipy_pdf = scipy_uniform.pdf(x)
    assert isinstance(pdf, np.ndarray)
    assert np.allclose(pdf, scipy_pdf, equal_nan=True)


@given(st.floats(allow_nan=True))
def test_uniform_icdf_float(y):
    icdf = uniform.icdf(y)
    scipy_icdf = scipy_uniform.ppf(y)
    assert isinstance(icdf, float)
    assert np.allclose(icdf, scipy_icdf, equal_nan=True)


@given(array_shapes(min_dims=1, min_side=1))
def test_uniform_icdf_array(size):
    y = np.random.normal(0, 1, size=size)
    icdf = uniform.icdf(y)
    scipy_icdf = scipy_uniform.ppf(y)
    assert isinstance(icdf, np.ndarray)
    assert np.allclose(icdf, scipy_icdf, equal_nan=True)


@given(st.floats(0, 1))
def test_uniform_icdf_cdf(x):
    """Reconstruct x as x = icdf(cdf(x))"""
    y = uniform.icdf(uniform.cdf(x))
    assert np.allclose(x, y, equal_nan=True)
