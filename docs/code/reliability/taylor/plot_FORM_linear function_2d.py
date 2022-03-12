"""

FORM - Linear function - Two Random variables
==============================================

In the second example we have the limit state to be a linear function of two (:math:`d=2`) independent Gaussian random
variables
"""

#%% md
#
# :math:`g(U) = -\frac{1}{\sqrt{d}}\sum_{i=1}^{d} u_i + \beta`
#
# The probability of failure in this case is :math:`P(F) ≈ 10^{−3}` for :math:`β = 3.0902`
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


dist1 = Normal(loc=0., scale=1.)
dist2 = Normal(loc=0., scale=1.)
RunModelObject2 = RunModel(model_script='local_pfn.py', model_object_name="example2", vec=False, ntasks=3)
Z = FORM(distributions=[dist1, dist2], runmodel_object=RunModelObject2)
Z.run()

# print results
print('Design point in standard normal space: %s' % Z.DesignPoint_U)
print('Design point in original space: %s' % Z.DesignPoint_X)
print('Hasofer-Lind reliability index: %s' % Z.beta_form)
print('FORM probability of failure: %s' % Z.failure_probability)

shutil.rmtree(RunModelObject2.model_dir)
