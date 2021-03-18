from UQpy.Transformations import *
import numpy as np
from UQpy.Distributions import Gamma, Lognormal

# Unit tests

def test_1():
    """
    Test Nataf class
    """
    dist1 = Gamma(4.0, loc=0.0, scale=1.0)
    dist2 = Lognormal(s=2., loc=0., scale=np.exp(1))
    Rx = np.array([[1.0, 0.9], [0.9, 1.0]])
    nataf_obj = Nataf(dist_object=[dist1,dist2], corr_x=Rx)
    c = np.array_equal(nataf_obj.corr_z.round(8), np.array([[1., 0.99996829],[0.99996829, 1.]]))
    assert c

