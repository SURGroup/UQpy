# Test the Correlation module

from UQpy.Transformations import Correlate
import numpy as np
import pytest


def test_samples():
    samples_z = np.array([[0.3, 0.2], [0.2, 2.4]])
    rz = np.array([[1.0, 0.8], [0.8, 1.0]])
    ntf_obj = Correlate(samples_u=samples_z, corr_z=rz)
    np.testing.assert_allclose(ntf_obj.samples_z, [[0.3, 0.36], [0.2, 1.6]], rtol=1e-09)


def test_samples_u():
    samples_z = np.array([[0.3, 0.2], [0.2, 2.4]])
    with pytest.raises(Exception):
        assert Correlate(samples_u=samples_z)


def test_corr_z():
    rz = np.array([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(Exception):
        assert Correlate(corr_z=rz)
