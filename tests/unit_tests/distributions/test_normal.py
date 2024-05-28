import numpy as np
from scipy import stats
from UQpy.distributions import Normal
import hypothesis.strategies as st
from hypothesis import given
from hypothesis.extra.numpy import array_shapes

normal = Normal()
scipy_normal = stats.norm()


@given(st.floats(allow_nan=True))
def test_normal_pdf_float(x):
    """Test custom implementation of normal pdf on float inputs. Should return flosa"""
    pdf = normal.pdf(x)
    scipy_pdf = scipy_normal.pdf(x)
    assert isinstance(pdf, float)
    assert np.allclose(pdf, scipy_pdf, equal_nan=True)


@given(array_shapes(min_dims=1, min_side=1))
def test_normal_pdf_array(size):
    x = np.random.normal(0, 1, size=size)
    pdf = normal.pdf(x)
    scipy_pdf = scipy_normal.pdf(x)
    assert isinstance(pdf, np.ndarray)
    assert np.allclose(pdf, scipy_pdf, equal_nan=True)


@given(st.floats(allow_nan=True))
def test_normal_cdf_float(x):
    pdf = normal.pdf(x)
    scipy_pdf = scipy_normal.pdf(x)
    assert isinstance(pdf, float)
    assert np.allclose(pdf, scipy_pdf, equal_nan=True)


@given(array_shapes(min_dims=1, min_side=1))
def test_normal_cdf_array(size):
    x = np.random.normal(0, 1, size=size)
    pdf = normal.pdf(x)
    scipy_pdf = scipy_normal.pdf(x)
    assert isinstance(pdf, np.ndarray)
    assert np.allclose(pdf, scipy_pdf, equal_nan=True)


@given(st.floats(min_value=1e-13))
def test_normal_icdf_float(y):
    """Note: icdf deviates from scipy ppf for values of y between 0 and 1e-13"""
    icdf = normal.icdf(y)
    scipy_icdf = scipy_normal.ppf(y)
    assert isinstance(icdf, float)
    assert np.allclose(icdf, scipy_icdf, equal_nan=True)


@given(array_shapes(min_dims=1, min_side=1))
def test_normal_icdf_array(size):
    y = np.random.normal(0, 1, size=size)
    icdf = normal.icdf(y)
    scipy_icdf = scipy_normal.ppf(y)
    assert isinstance(icdf, np.ndarray)
    assert np.allclose(icdf, scipy_icdf, equal_nan=True)


@given(st.floats(-7, 7, allow_nan=False, allow_infinity=False))
def test_normal_cdf_icdf(x):
    """Reconstruct x as x = icdf(cdf(x))
    Note: For +/- 7 standard deviations, UQpy and SciPy accurately reconstruct x.
    At 8 standard deviations, both UQpy and scipy.stats.norm() begin to divergence from the correct answer
    """
    y = normal.icdf(normal.cdf(x))
    assert np.allclose(x, y)
