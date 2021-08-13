from UQpy.Surrogates import SROM
from UQpy.SampleMethods import RectangularStrata, RectangularSTS
from UQpy.Distributions import Gamma
import pytest
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

marginals = [Gamma(a=2., loc=1., scale=3.), Gamma(a=2., loc=1., scale=3.)]
strata = RectangularStrata(nstrata=[4, 4])
x = RectangularSTS(dist_object=marginals, strata_object=strata, nsamples_per_stratum=1, random_state=1)


def test_run1():
    y = SROM(samples=x.samples, target_dist_object=marginals, moments=np.array([[6., 6.], [54., 54.]]),
             weights_errors=[1, 0.2, 0.1], verbose=True)
    y.run(properties=[True, True, True, True])
    tmp = np.round(y.sample_weights, 3) == np.array([0., 0., 0.076, 0.182, 0.082, 0.059, 0.113, 0.055, 0.074, 0.03,
                                                     0.062, 0.062, 0.06, 0.145, 0., 0.])
    assert tmp.all()


def test_run2():
    y = SROM(samples=x.samples, target_dist_object=marginals, moments=np.array([[6., 6.], [54., 54.]]))
    y.run(properties=[True, True, True, False])
    tmp = np.round(y.sample_weights, 3) == np.array([0.051, 0.023, 0.084, 0.05, 0.108, 0.071, 0.054, 0.061, 0.006,
                                                     0.065, 0.079, 0.138, 0.032, 0.046, 0.039, 0.092])
    assert tmp.all()


def test_run3():
    y = SROM(samples=x.samples.tolist(), target_dist_object=marginals, moments=[[6., 6.], [54., 54.]],
             weights_errors=np.array([1, 0.2, 0.1]), properties=[True, True, True, True],
             weights_distribution=np.ones(shape=(x.samples.shape[0], x.samples.shape[1])).tolist(),
             weights_moments=np.reciprocal(np.square([[6., 6.], [54., 54.]])).tolist(),
             weights_correlation=np.identity(x.samples.shape[1]).tolist(),
             correlation=np.identity(x.samples.shape[1]).tolist())
    tmp = np.round(y.sample_weights, 3) == np.array([0.051, 0.023, 0.084, 0.05, 0.108, 0.071, 0.054, 0.061, 0.006,
                                                     0.065, 0.079, 0.138, 0.032, 0.046, 0.039, 0.092])
    assert tmp.all()


def test_run4():
    y = SROM(samples=x.samples, target_dist_object=marginals, moments=np.array([[6., 6.], [54., 54.]]))
    y.run(properties=[True, True, True, False], weights_errors=np.array([1, 0.2, 0.1]))
    tmp = np.round(y.sample_weights, 3) == np.array([0.051, 0.023, 0.084, 0.05, 0.108, 0.071, 0.054, 0.061, 0.006,
                                                     0.065, 0.079, 0.138, 0.032, 0.046, 0.039, 0.092])
    assert tmp.all()


def test_run5():
    y = SROM(samples=x.samples.tolist(), target_dist_object=marginals, moments=[[6., 6.], [54., 54.]],
             weights_errors=np.array([1, 0.2, 0.1]))
    y.run(properties=[True, True, True, True],
          weights_distribution=np.ones(shape=(x.samples.shape[0], x.samples.shape[1])).tolist(),
          weights_moments=np.reciprocal(np.square([[6., 6.], [54., 54.]])).tolist(),
          weights_correlation=np.identity(x.samples.shape[1]).tolist())
    tmp = np.round(y.sample_weights, 3) == np.array([0.051, 0.023, 0.084, 0.05, 0.108, 0.071, 0.054, 0.061, 0.006,
                                                     0.065, 0.079, 0.138, 0.032, 0.046, 0.039, 0.092])
    assert tmp.all()


def test_sample_type():
    """
        'samples' attribute should be a list or numpy array.
    """
    with pytest.raises(NotImplementedError):
        SROM(samples='x.samples', target_dist_object=marginals, moments=np.array([[6., 6.], [54., 54.]]),
             weights_errors=[1, 0.2, 0.1])


def test_target_distribution_type():
    """
        'target_dist_object' can't be None.
    """
    with pytest.raises(NotImplementedError):
        SROM(samples=x.samples, target_dist_object=None, moments=np.array([[6., 6.], [54., 54.]]),
             weights_errors=[1, 0.2, 0.1])


def test_target_distribution_type1():
    """
        'target_dist_object' should be a distribution object or list of distribution object.
    """
    with pytest.raises(TypeError):
        SROM(samples=x.samples, target_dist_object=['a', 'b'], moments=np.array([[6., 6.], [54., 54.]]),
             weights_errors=[1, 0.2, 0.1])


def test_moments():
    """
        'moments' attribute can't be None.
    """
    with pytest.raises(NotImplementedError):
        SROM(samples=x.samples, target_dist_object=marginals, weights_errors=[1, 0.2, 0.1],
             properties=[True, True, True, True])


def test_moments1():
    """
        Verify the check on the shape of 'moments' attribute.
    """
    with pytest.raises(NotImplementedError):
        y = SROM(samples=x.samples, target_dist_object=marginals, weights_errors=[1, 0.2, 0.1],
                 moments=np.array([[6., 6., 6.], [54., 54., 54.]]))
        y.run()


def test_moments2():
    """
        Verify the check on the shape of 'moments' attribute.
    """
    with pytest.raises(NotImplementedError):
        y = SROM(samples=x.samples, target_dist_object=marginals, weights_errors=[1, 0.2, 0.1],
                 moments=np.array([[6., 6.]]))
        y.run()


def test_moments3():
    """
        Verify the check on the shape of 'moments' attribute.
    """
    with pytest.raises(NotImplementedError):
        y = SROM(samples=x.samples, target_dist_object=marginals, weights_errors=[1, 0.2, 0.1],
                 properties=[True]*4, moments=np.array([[6., 6.]]))
        y.run()


def test_moments4():
    """
        'moments' attribute: If first moment is not used, attribute is modified such that second moment is in
        the second row.
    """
    moments = np.array([[54., 54.]])
    y = SROM(samples=x.samples, target_dist_object=marginals, weights_errors=[1, 0.2, 0.1],
             properties=[True, False, True, False], moments=moments)
    assert (y.moments == np.concatenate((np.ones(shape=(1, y.dimension)), moments))).all()


def test_weights_error():
    """
        'weights_error' attribute should be None, list or numpy array.
    """
    with pytest.raises(NotImplementedError):
        SROM(samples=x.samples, target_dist_object=marginals, weights_errors='[1, 0.2, 0.1]',
             properties=[False, True, False, False], moments=np.array([[54., 54.]]))


def test_weights_distribution():
    """
        'weights_distribution' attribute should be None, list or numpy array.
    """
    with pytest.raises(NotImplementedError):
        SROM(samples=x.samples, target_dist_object=marginals, weights_errors=[1, 0.2, 0.1],
             properties=[False, True, False, False], moments=np.array([[54., 54.]]), weights_distribution='a')


def test_weights_distribution_shape():
    """
        Verify the shape of 'weights_distribution' attribute.
    """
    weights_distribution = np.ones(shape=(1, x.samples.shape[1])).tolist()
    y = SROM(samples=x.samples, target_dist_object=marginals, weights_errors=[1, 0.2, 0.1],
             properties=[False, True, False, False], moments=np.array([[54., 54.]]),
             weights_distribution=weights_distribution)
    expected_value = weights_distribution * np.ones(shape=(y.samples.shape[0], y.dimension))
    assert (y.weights_distribution == expected_value).all()


def test_weights_distribution_shape2():
    """
        Verify the shape of 'weights_distribution' attribute.
    """
    with pytest.raises(NotImplementedError):
        SROM(samples=x.samples, target_dist_object=marginals, weights_errors=[1, 0.2, 0.1],
             properties=[False, True, False, False], moments=np.array([[54., 54.]]),
             weights_distribution=np.ones(shape=(3, x.samples.shape[1])).tolist())


def test_weights_moments():
    """
        'weights_moments' attribute should be None, a list or numpy array.
    """
    with pytest.raises(NotImplementedError):
        SROM(samples=x.samples, target_dist_object=marginals, weights_errors=[1, 0.2, 0.1],
             properties=[False, True, False, False], moments=np.array([[54., 54.]]),
             weights_moments='a')


def test_weights_moments_shape():
    """
        Verify the shape of the 'weights_moments' attribute.
    """
    with pytest.raises(NotImplementedError):
        y = SROM(samples=x.samples.tolist(), target_dist_object=marginals, moments=[[6., 6.], [54., 54.]],
                 weights_errors=np.array([1, 0.2, 0.1]))
        y.run(properties=[True, True, True, True],
              weights_moments=np.reciprocal(np.square([[6., 6., 6.], [54., 54., 54.]])).tolist())


def test_weights_correlation():
    """
        'weights_correlation' attribute should be None, a list or numpy array.
    """
    with pytest.raises(NotImplementedError):
        SROM(samples=x.samples, target_dist_object=marginals, weights_errors=[1, 0.2, 0.1],
             properties=[False, True, False, False], moments=np.array([[54., 54.]]),
             weights_correlation='a')


def test_weights_correlation_shape():
    """
        Verify shape of the 'weights_correlation' attribute.
    """
    with pytest.raises(NotImplementedError):
        SROM(samples=x.samples, target_dist_object=marginals, weights_errors=[1, 0.2, 0.1],
             properties=[False, True, False, False], moments=np.array([[54., 54.]]),
             weights_correlation=np.identity(3))
