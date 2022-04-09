"""

4. SORM - Nonlinear function - Two Random variables
=====================================================

In the fourth example we have the limit state to be a nonlinear function of two (:math:`d=2`) random variables

.. math:: g(X_1, X_2) = X_1X_1 - 80

where :math:`X_1` follows a normal distribution with mean :math:`\mu_{X_1}=20` and standard deviation
:math:`\sigma_{X_1}=7` and :math:`X_2` follows a lognormal distribution with mean :math:`\mu_{X_2}=7` and
standard deviation :math:`\sigma_{X_2}=1.4`.
"""

#%% md
#
# Initially we have to import the necessary modules.

#%%

import shutil

import numpy as np
from UQpy.run_model.RunModel import RunModel
from UQpy.run_model.model_execution.PythonModel import PythonModel
from UQpy.distributions import Normal
from UQpy.reliability import FORM
from UQpy.reliability import SORM
from UQpy.distributions import Lognormal

m0 = 7
v0 = 1.4
mu = np.log(m0) - np.log(np.sqrt(1 + (v0 / m0) ** 2))
scale = np.exp(mu)
s = np.sqrt(np.log(1 + (v0 / m0) ** 2))
loc_ = 0.0

dist1 = Normal(loc=20., scale=2)
dist2 = Lognormal(s=s, loc=0.0, scale=scale)
model = PythonModel(model_script='pfn.py', model_object_name="example4",)
RunModelObject4 = RunModel(model=model)
form = FORM(distributions=[dist1, dist2], runmodel_object=RunModelObject4)
form.run()
Q0 = SORM(form_object=form)


# print results
print('SORM probability of failure: %s' % Q0.failure_probability)

