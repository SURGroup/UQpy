from UQpy.sampling.LatinHypercubeSampling import *
from UQpy.distributions.collection.Uniform import *
from UQpy.sampling.latin_hypercube_criteria.Random import Random
from UQpy.sampling.latin_hypercube_criteria.Centered import *
from UQpy.sampling.latin_hypercube_criteria.MaxiMin import *
from UQpy.sampling.latin_hypercube_criteria.MinCorrelation import *
import numpy as np

distribution = Uniform(1, 1)
distribution1 = Uniform(1, 2)
distribution2 = Uniform(3, 4)


def test_lhs_random_criterion():
    random_criterion = Random(random_state=1)
    latin_hypercube_sampling = \
        LatinHypercubeSampling(distributions=distribution, samples_number=2,
                               verbose=True, criterion=random_criterion)
    actual_samples = latin_hypercube_sampling.samples.flatten()
    expected_samples = np.array([[1.208511], [1.86016225]]).flatten()
    np.testing.assert_allclose(expected_samples, actual_samples, rtol=1e-6)


def test_lhs_centered_criterion():
    centered_criterion = Centered(random_state=1)
    latin_hypercube_sampling = \
        LatinHypercubeSampling(distributions=distribution, samples_number=2,
                               verbose=True, criterion=centered_criterion)
    actual_samples = latin_hypercube_sampling.samples.flatten()
    expected_samples = np.array([[1.25], [1.75]]).flatten()
    np.testing.assert_allclose(expected_samples, actual_samples, rtol=1e-6)


def test_lhs_maximin_criterion():
    maximin_criterion = MaxiMin(random_state=1)
    latin_hypercube_sampling = \
        LatinHypercubeSampling(distributions=distribution, samples_number=2,
                               verbose=True, criterion=maximin_criterion)
    actual_samples = latin_hypercube_sampling.samples.flatten()
    expected_samples = np.array([[1.208511], [1.86016225]]).flatten()
    np.testing.assert_allclose(expected_samples, actual_samples, rtol=1e-6)
