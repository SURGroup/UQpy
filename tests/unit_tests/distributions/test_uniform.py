import pytest
import numpy as np
from scipy import stats
from UQpy.distributions import Uniform
from hypothesis import given
from hypothesis.extra.numpy import array_shapes


class TestUniform:

    uniform = Uniform()
    scipy_uniform = stats.uniform()

    def test_pdf_nan(self):
        """Consistent with scipy, pdf(NaN)=NaN"""
        assert np.isnan(self.uniform.pdf(np.nan))

    def test_pdf_infinity(self):
        """Consistent with scipy, pdf(inf)=0 and pdf(-inf)=0"""
        assert np.isclose(self.uniform.pdf(np.inf), 0.0)
        assert np.isclose(self.uniform.pdf(-np.inf), 0.0)

    @given(array_shapes(min_dims=1, min_side=1))
    def test_pdf_shape(self, shape):
        """Test if the output array matches the shape of the input array"""
        x = np.zeros(shape)
        assert x.shape == self.uniform.pdf(x).shape

    def test_pdf_values(self):
        """Test if UQpy pdf matches scipy pdf for x in [-1, 2]"""
        x = np.linspace(-1, 2, num=100)
        assert np.allclose(self.uniform.pdf(x), self.scipy_uniform.pdf(x))

    def test_cdf_nan(self):
        """Consistent with scipy cdf(NaN)=NaN"""
        assert np.isnan(self.uniform.cdf(np.nan))

    @pytest.mark.parametrize("test_input,expected", [(np.inf, 1.0), (-np.inf, 0.0)])
    def test_cdf_infinity(self, test_input, expected):
        """Consistent with scipy cdf(inf)=1.0, cdf(-inf)=0.0"""
        assert self.uniform.cdf(test_input) == expected

    @given(array_shapes(min_dims=1, min_side=1))
    def test_cdf_shape(self, shape):
        """Test if the output array matches the shape of the input array"""
        x = np.zeros(shape)
        assert x.shape == self.uniform.cdf(x).shape

    def test_cdf_values(self):
        """Test if UQpy cdf matches scipy cdf on x in [-1, 2]"""
        x = np.linspace(-1, 2, num=100)
        assert np.allclose(self.uniform.cdf(x), self.scipy_uniform.cdf(x))

    def test_icdf_nan(self):
        """Consistent with scipy icdf(NaN)=NaN"""
        assert np.isnan(self.uniform.icdf(np.nan))

    def test_icdf_infinity(self):
        """Consistent with scipy icdf(inf)=NaN and icdf(-inf)=NaN"""
        assert np.isnan(self.uniform.icdf(np.inf))
        assert np.isnan(self.uniform.icdf(-np.inf))

    @given(array_shapes(min_dims=1, min_side=1))
    def test_icdf_shape(self, shape):
        """Test if the output array has the same shape as the input array"""
        x = np.zeros(shape)
        assert x.shape == self.uniform.icdf(x).shape

    def test_icdf_values(self):
        """Test if UQpy icdf matches scipy ppf for y in [-0.1, 1.1]. Note outside of [0, 1] both return NaN"""
        y = np.linspace(-0.1, 1.1, num=100)
        assert np.allclose(
            self.uniform.icdf(y), self.scipy_uniform.ppf(y), equal_nan=True
        )

    def test_cdf_icdf_reconstruction(self):
        """Reconstruct x as x = icdf(cdf(x)). Note that this only works where cdf(x) is invertible on [0,1]"""
        x = np.linspace(0, 1, num=100)
        y = self.uniform.cdf(x)
        x_reconstruction = self.uniform.icdf(y)
        assert np.allclose(x, x_reconstruction)
