import pytest
import numpy as np
from scipy import stats
from UQpy.distributions import Normal
from hypothesis import given
from hypothesis.extra.numpy import array_shapes


class TestNormal:

    normal = Normal()
    scipy_normal = stats.norm()

    def test_pdf_nan(self):
        """Consistent with scipy, pdf(NaN)=NaN"""
        assert np.isnan(self.normal.pdf(np.nan))

    def test_pdf_infinity(self):
        """Consistent with scipy pdf(inf)=0.0, pdf(-inf)=0.0"""
        assert np.isclose(self.normal.pdf(np.inf), 0.0)
        assert np.isclose(self.normal.pdf(-np.inf), 0.0)

    @given(array_shapes(min_dims=1, min_side=1))
    def test_pdf_shape(self, shape):
        """Test the output array matches the shape of the input array"""
        x = np.zeros(shape)
        assert x.shape == self.normal.pdf(x).shape

    def test_pdf_values(self):
        """Test if UQpy pdf matches SciPy pdf for x in [-10, 10]"""
        x = np.linspace(-10, 10, num=1_000)
        assert np.allclose(self.normal.pdf(x), self.scipy_normal.pdf(x))

    def test_cdf_nan(self):
        """Consistent with scipy, cdf(NaN)=NaN"""
        assert np.isnan(self.normal.cdf(np.nan))

    @pytest.mark.parametrize("test_input,expected", [(np.inf, 1.0), (-np.inf, 0.0)])
    def test_cdf_infinity(self, test_input, expected):
        """Consistent with scipy cdf(inf)=1.0, cdf(-inf)=0.0"""
        assert self.normal.cdf(test_input) == expected

    @given(array_shapes(min_dims=1, min_side=1))
    def test_cdf_shape(self, shape):
        """Test if output array matches the shape of the input array"""
        x = np.zeros(shape)
        assert x.shape == self.normal.cdf(x).shape

    def test_cdf_values(self):
        """Test if UQpy cdf matches scipy cdf for x in [-10, 10]"""
        x = np.linspace(-10, 10, num=100)
        assert np.allclose(self.normal.cdf(x), self.scipy_normal.cdf(x))

    def test_icdf_nan(self):
        """Consistent with scipy, icdf(NaN) = NaN"""
        assert np.isnan(self.normal.icdf(np.nan))

    def test_icdf_infinity(self):
        """Consistent with scipy icdf(inf)=Nan and icdf(-inf)=NaN"""
        assert np.isnan(self.normal.icdf(np.inf))
        assert np.isnan(self.normal.icdf(-np.inf))

    def test_icdf_zero_one(self):
        """Consistent with scipy, icdf(0)=-inf and icdf(1)=inf"""
        assert self.normal.icdf(0) == -np.inf
        assert self.normal.icdf(1) == np.inf

    def test_icdf_values(self):
        """Test if UQpy icdf matches scipy ppf for y in [-0.1, 1.1]. Note outside [0, 1] both functions return NaN"""
        y = np.linspace(-0.1, 1.1, num=100)
        assert np.allclose(self.normal.icdf(y), self.scipy_normal.ppf(y), equal_nan=True)

    def test_cdf_icdf_reconstruction(self):
        """Reconstruct x as x = icdf(cdf(x))
        Note: For +/- 7 standard deviations, UQpy and SciPy accurately reconstruct x.
        At 8 standard deviations, both UQpy and scipy.stats.norm() begin to divergence from the correct answer
        """
        x = np.linspace(-7, 7, num=100)
        y = self.normal.cdf(x)
        x_reconstruction = self.normal.icdf(y)
        assert np.allclose(x, x_reconstruction)
