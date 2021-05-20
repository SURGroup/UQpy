from UQpy.Surrogates import SROM
from UQpy.SampleMethods import RectangularStrata, RectangularSTS
from UQpy.Distributions import Gamma
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

marginals = [Gamma(a=2., loc=1., scale=3.), Gamma(a=2., loc=1., scale=3.)]
strata = RectangularStrata(nstrata=[4, 4])
x = RectangularSTS(dist_object=marginals, strata_object=strata, nsamples_per_stratum=1, random_state=1)


def test_run1():
    y = SROM(samples=x.samples, target_dist_object=marginals, moments=np.array([[6., 6.], [54., 54.]]),
             weights_errors=[1, 0.2, 0.1])
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
