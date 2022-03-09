"""

FORM - Linear function - Three Random variables
==============================================

In the third example we have the limit state to be a linear function of three (:math:`d`=3) independent Gaussian random
variables

.. math:: g(X_1, X_2, X_3) = 6.2X_1 -X_2X_3^2

.. math:: \mu_{X_1}=20, \mu_{X_2}=5, \mu_{X_3}=4

.. math:: \sigma_{X_1}=3.5, \sigma_{X_2}=0.8, \sigma_{X_3}=0.4

The probability of failure in this case is :math:`P_f ≈ 0.079` for :math:`\beta = 1.413`
"""

#%% md
#
# Initially we have to import the necessary modules.

#%%

import shutil

import numpy as np
import matplotlib.pyplot as plt
from UQpy.RunModel import RunModel
from UQpy.distributions import Normal
from UQpy.sampling import MonteCarloSampling
from UQpy.reliability import FORM, SORM
from UQpy.reliability import SORM
from UQpy.distributions import Lognormal, Gamma


dist1 = Normal(loc=20., scale=3.5)
dist2 = Normal(loc=5., scale=0.8)
dist3 = Normal(loc=4., scale=0.4)
RunModelObject3 = RunModel(model_script='pfn.py', model_object_name="example3", vec=False, ntasks=3)
Z0 = FORM(distributions=[dist1, dist2, dist3], runmodel_object=RunModelObject3)
Z0.run()

print('Design point in standard normal space: %s' % Z0.DesignPoint_U)
print('Design point in original space: %s' % Z0.DesignPoint_X)
print('Hasofer-Lind reliability index: %s' % Z0.beta_form)
print('FORM probability of failure: %s' % Z0.failure_probability)

shutil.rmtree(RunModelObject3.model_dir)