from UQpy.sampling.MonteCarloSampling import MonteCarloSampling
import numpy as np
import matplotlib.pyplot as plt
import time
# import pytest
# import os
from UQpy.distributions.collection.Normal import Normal

def test_mcs_dummy():
    dist1 = Normal(loc=0., scale=1.)
    dist2 = Normal(loc=0., scale=1.)


    x = MonteCarloSampling(distributions=[dist1, dist2], samples_number=5, random_state=np.random.RandomState(123))

    a=1

