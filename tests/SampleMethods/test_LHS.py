import numpy as np
import pytest
from UQpy.SampleMethods import LHS
from UQpy.Distributions import Uniform, JointInd
from scipy.spatial.distance import pdist

dist1 = Uniform(loc=0., scale=1.)
dist2 = Uniform(loc=0., scale=1.)
joint_dist = JointInd(marginals=[dist1, dist2])

x1a = LHS(dist_object=dist1, criterion='maximin', random_state=123, nsamples=5, verbose=True)

x1b = LHS(dist_object=joint_dist, criterion='correlate', nsamples=5, random_state=123, verbose=True)

x1c = LHS(dist_object=[dist1, dist2], nsamples=5, random_state=np.random.default_rng(789), verbose=True)

x1d = LHS(dist_object=dist1, criterion='centered', random_state=123, nsamples=5, verbose=True)

x1e = LHS(dist_object=dist1, criterion='centered', nsamples=5, verbose=True)

cut = np.linspace(0, 1, 5 + 1)
x1f = LHS(dist_object=dist1, criterion=LHS.centered, random_state=123, nsamples=5, a=cut[:5], b=cut[1:6])

x1g = LHS(dist_object=[dist1, dist2], nsamples=5)


def d_func(x): return pdist(x, metric='euclidean')


x1h = LHS(dist_object=[dist1, dist2], criterion='maximin', nsamples=5, random_state=789, metric=d_func)


def test_samples1():
    """ Check the samples attribute, when dist_object is a distribution class object and criterion is 'maximin'."""
    expected_samples = np.array([0.44, 0.84, 0.14, 0.64, 0.21])
    assert (x1a.samples.round(2) == expected_samples).all()


def test_samples2():
    """ Check the samples attribute, when dist_object is a jointInd class object and criterion is 'correlate'."""
    expected_samples = np.array([[0.21, 0.76], [0.14, 0.16], [0.64, 0.38], [0.44, 0.98], [0.84, 0.46]])
    assert (x1b.samples.round(2) == expected_samples).all()


def test_samples3():
    """ Check the samples attribute, when dist_object is a list of distribution class object, criterion is
    'random'."""
    expected_samples = np.array([[0.24, 0.49], [0.63, 0.82], [0.02, 0.12], [0.53, 0.22], [0.8 , 0.75]])
    assert (x1c.samples.round(2) == expected_samples).all()


def test_samples4():
    """ Check the samples attribute, when dist_object is a list of distribution class object and criterion is
    'centered'."""
    expected_samples = np.array([0.5, 0.9, 0.1, 0.7, 0.3])
    assert (x1d.samples.round(2) == expected_samples).all()


def test_samples5():
    """ Check the samples attribute, when dist_object is a list of distribution class object, criterion is
    'maximin' and metric is callable."""
    expected_samples = np.array([[0.24, 0.22], [0.8 , 0.49], [0.63, 0.12], [0.02, 0.82], [0.53, 0.75]])
    assert (x1h.samples.round(2) == expected_samples).all()


def test_iterations():
    """Check iteration attribute, when 'correlat' criterion is used"""
    with pytest.raises(ValueError):
        x1c.correlate(samples=0, iterations='ab')


def test_dist_object():
    """Validate dist_object, when dist_object is a distribution object."""
    with pytest.raises(TypeError):
        LHS(dist_object='abc', nsamples=2)


def test_dist_object2():
    """Validate dist_object, when dist_object is not a list of distribution object."""
    with pytest.raises(TypeError):
        LHS(dist_object=['abc'], nsamples=2)


def test_random_state1():
    """Check if random_state attribute is an integer, np.random.RandomState object or None, when dist_object is a
    distribution class object."""
    with pytest.raises(TypeError):
        LHS(dist_object=dist1, random_state='abc', nsamples=2)


def test_criterion():
    """Check if criterion attribute is not a string/callable."""
    with pytest.raises(NotImplementedError):
        LHS(dist_object=dist1, criterion='abc', nsamples=2)


def test_criterion2():
    """Check if criterion attribute is not a string/callable."""
    with pytest.raises(ValueError):
        LHS(dist_object=dist1, criterion=1, nsamples=2)


def test_nsamples():
    """Check if nsamples attribute is not an integer. """
    with pytest.raises(ValueError):
        LHS(dist_object=dist1, nsamples='abc')


def test_nsamples2():
    """Check if nsamples attribute is not an integer, check for nsamples inside 'run' method."""
    with pytest.raises(ValueError):
        x1h.run(nsamples='abc')


def test_maximin_metric():
    """Check if metric attribute is valid, when criterion is a 'maximin'."""
    with pytest.raises(NotImplementedError):
        LHS(dist_object=dist1, criterion='maximin', random_state=123, nsamples=5, metric='abc')


def test_maximin_metric1():
    """Check if metric attribute is valid, when criterion is a 'maximin'."""
    with pytest.raises(ValueError):
        LHS(dist_object=dist1, criterion='maximin', random_state=123, nsamples=5, metric=1)


def test_maximin_iterations():
    """Check if metric attribute is valid, when criterion is a 'maximin'."""
    with pytest.raises(ValueError):
        LHS(dist_object=dist1, criterion='maximin', random_state=123, nsamples=5, iterations='abc')

