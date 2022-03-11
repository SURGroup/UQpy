"""

Function with nonlinearities / parameter dependencies
=================================================================

.. math:: Y = h(X) = 0.01 X_{1} + 1.0 X_{2} + 0.4 X_{3}^{2} + X_{4} X_{5}

ranking of input parameters:

- :math:`X_{1}` is non-influential
- :math:`X_{2}` is influential, linear/additive effect (expect large :math:`\mu^{\star}` and small :math:`\sigma`)
- :math:`X_{3}` is somewhat influential, nonlinear effect,
- :math:`X_{4}, X_{5}` are influential with dependence

"""

#%% md
#
# Initially we have to import the necessary modules.

#%%
import shutil

from UQpy.RunModel import RunModel
from UQpy.distributions import Uniform
from UQpy.sensitivity import MorrisSensitivity
import numpy as np
import matplotlib.pyplot as plt

#%% md
#
# Set-up problem with g-function.

#%%

var_names = ['X{}'.format(i) for i in range(5)]
runmodel_object = RunModel(
    model_script='pfn.py', model_object_name='fun2_sensitivity', var_names=var_names, vec=True)

dist_object = [Uniform(), ] * 5


sens = MorrisSensitivity(runmodel_object=runmodel_object,
                         distributions=dist_object,
                         n_levels=20, maximize_dispersion=True)
sens.run(n_trajectories=10)


fig, ax = plt.subplots(figsize=(5, 3.5))
ax.scatter(sens.mustar_indices, sens.sigma_indices, s=60)
for i, (mu, sig) in enumerate(zip(sens.mustar_indices, sens.sigma_indices)):
    ax.text(x=mu + 0.01, y=sig + 0.01, s='X{}'.format(i + 1), fontsize=14)
ax.set_xlabel(r'$\mu^{\star}$', fontsize=18)
ax.set_ylabel(r'$\sigma$', fontsize=18)
# ax.set_title('Morris sensitivity indices', fontsize=16)
plt.show()

shutil.rmtree(runmodel_object.model_dir)
