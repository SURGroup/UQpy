from UQpy import Centered, MaxiMin
from UQpy.sampling.stratified_sampling.LatinHypercubeSampling import *
from UQpy.distributions.collection.Uniform import *
from UQpy.sampling.stratified_sampling.latin_hypercube_criteria.MinCorrelation import *
import numpy as np

distribution = Uniform(1, 1)
distribution1 = Uniform(1, 2)
distribution2 = Uniform(3, 4)

def test_lhs_random_criterion():
    random_criterion = Random()
    latin_hypercube_sampling = \
        LatinHypercubeSampling(distributions=distribution, nsamples=2,
                               criterion=random_criterion, random_state=1)
    actual_samples = latin_hypercube_sampling.samples.flatten()
    expected_samples = np.array([[1.208511], [1.86016225]]).flatten()
    np.testing.assert_allclose(expected_samples, actual_samples, rtol=1e-6)


def test_lhs_centered_criterion():
    centered_criterion = Centered()
    latin_hypercube_sampling = \
        LatinHypercubeSampling(distributions=distribution, nsamples=2,
                               criterion=centered_criterion, random_state=1)
    actual_samples = latin_hypercube_sampling.samples.flatten()
    expected_samples = np.array([[1.25], [1.75]]).flatten()
    np.testing.assert_allclose(expected_samples, actual_samples, rtol=1e-6)


def test_lhs_maximin_criterion():
    maximin_criterion = MaxiMin()
    latin_hypercube_sampling = \
        LatinHypercubeSampling(distributions=distribution, nsamples=2,
                               criterion=maximin_criterion, random_state=1)
    actual_samples = latin_hypercube_sampling.samples.flatten()
    expected_samples = np.array([[1.208511], [1.86016225]]).flatten()
    np.testing.assert_allclose(expected_samples, actual_samples, rtol=1e-6)

def test_lhs_maximin_criterion_joint_independent():
    maximin_criterion = MaxiMin()
    latin_hypercube_sampling = \
        LatinHypercubeSampling(distributions=JointIndependent(marginals=[distribution, distribution1]),
                               nsamples=2, criterion=maximin_criterion, random_state=1)
    actual_samples = latin_hypercube_sampling.samples.flatten()
    expected_samples = np.array([1.86016225, 1.00011437, 1.208511,   2.30233257]).flatten()
    np.testing.assert_allclose(expected_samples, actual_samples, rtol=1e-6)


from scipy.spatial.distance import pdist

dist1 = Uniform(loc=0., scale=1.)
dist2 = Uniform(loc=0., scale=1.)
joint_dist = JointIndependent(marginals=[dist1, dist2])
x1b = LatinHypercubeSampling(distributions=joint_dist, criterion=MinCorrelation(), nsamples=5, random_state=123)

x1d = LatinHypercubeSampling(distributions=dist1, criterion=Centered(), random_state=123, nsamples=5)

x1e = LatinHypercubeSampling(distributions=dist1, criterion=Centered(), nsamples=5)
x1g = LatinHypercubeSampling(distributions=[dist1, dist2], nsamples=5)


def d_func(x): return pdist(x, metric='euclidean')


x1h = LatinHypercubeSampling(distributions=[dist1, dist2], criterion=MaxiMin(), nsamples=5, random_state=789)


def test_samples2():
    """ Check the samples attribute, when dist_object is a jointInd class object and criterion is 'correlate'."""
    expected_samples = np.array([[0.94, 0.54], [0.26, 0.08], [0.45, 0.88], [0.14, 0.7 ], [0.71, 0.4 ]])
    assert (x1b.samples.round(2) == expected_samples).all()


def test_samples4():
    """ Check the samples attribute, when dist_object is a list of distribution class object and criterion is
    'centered'."""
    expected_samples = np.array([0.1, 0.9, 0.5, 0.3, 0.7])
    assert (x1d.samples.round(2) == expected_samples).all()


def test_samples5():
    """ Check the samples attribute, when dist_object is a list of distribution class object, criterion is
    'maximin' and metric is callable."""
    expected_samples = np.array([[0.56, 0.2 ], [0.25, 0.62], [1.  , 0.4 ], [0.72, 0.91], [0.06, 0.15]])
    assert (x1h.samples.round(2) == expected_samples).all()
