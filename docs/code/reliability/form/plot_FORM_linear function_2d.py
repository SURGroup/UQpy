"""

1. FORM - Linear function - Two Random variables
===================================================

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

from UQpy.run_model.RunModel_New import RunModel_New
from UQpy.run_model.model_execution.PythonModel import PythonModel
from UQpy.distributions import Normal
from UQpy.reliability import FORM

dist1 = Normal(loc=0., scale=1.)
dist2 = Normal(loc=0., scale=1.)

model = PythonModel(model_script='pfn.py', model_object_name="example2")
RunModelObject2 = RunModel_New(model=model)

Z = FORM(distributions=[dist1, dist2], runmodel_object=RunModelObject2)
Z.run()

# print results
print('Design point in standard normal space: %s' % Z.DesignPoint_U)
print('Design point in original space: %s' % Z.DesignPoint_X)
print('Hasofer-Lind reliability index: %s' % Z.beta)
print('FORM probability of failure: %s' % Z.failure_probability)

