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
