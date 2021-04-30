# Test the Correlation module

from UQpy.Transformations import Decorrelate
import numpy as np
import pytest


def test_samples():
    samples_z = np.array([[0.3, 0.36], [0.2, 1.6]])
    rz = np.array([[1.0, 0.8], [0.8, 1.0]])
    ntf_obj = Decorrelate(samples_z=samples_z, corr_z=rz)
    np.testing.assert_allclose(
        ntf_obj.samples_u,
        [[0.3, 0.19999999999999998], [0.2, 2.4000000000000004]],
        rtol=1e-09,
    )


def test_samples_u():
    samples_z = np.array([[0.3, 0.36], [0.2, 1.6]])
    with pytest.raises(Exception):
        assert Decorrelate(samples_z=samples_z)


def test_corr_z():
    rz = np.array([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(Exception):
        assert Decorrelate(corr_z=rz)


def test_h():
    samples_z = np.array([[0.3, 0.36], [0.2, 1.6]])
    rz = np.array([[1.0, 0.8], [0.8, 1.0]])
    ntf_obj = Decorrelate(samples_z=samples_z, corr_z=rz)
    np.testing.assert_allclose(ntf_obj.H, [[1.0, 0.0], [0.8, 0.6]], rtol=1e-09)


def test_samples1():
    samples_z = np.array([[0.3, 0.36], [0.2, 1.6]])
    rz = np.array([[1.0, 0.8], [0.8, 1.0]])
    ntf_obj = Decorrelate(samples_z=samples_z, corr_z=rz)
    assert (ntf_obj.samples_z == np.array([[0.3, 0.36], [0.2, 1.6]])).all()


def test_corr_z_1():
    samples_z = np.array([[0.3, 0.36], [0.2, 1.6]])
    rz = np.array([[1.0, 0.8], [0.8, 1.0]])
    ntf_obj = Decorrelate(samples_z=samples_z, corr_z=rz)
    assert (ntf_obj.corr_z == np.array([[1.0, 0.8], [0.8, 1.0]])).all()
