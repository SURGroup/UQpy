"""

12-dimensional g-function
==============================================

To be compared with results from "An effective screening design for sensitivity analysis of large models",
Campolongo et al, 2007

"""

#%% md
#
# Initially we have to import the necessary modules.

#%%
import shutil

from UQpy.run_model.RunModel import RunModel
from UQpy.distributions import Uniform
from UQpy.sensitivity import MorrisSensitivity
import matplotlib.pyplot as plt

#%% md
#
# Set-up problem with g-function.

#%%

a_values = [0.001, 89.9, 5.54, 42.10, 0.78, 1.26, 0.04, 0.79, 74.51, 4.32, 82.51, 41.62]
na = len(a_values)

var_names = ['X{}'.format(i) for i in range(na)]
runmodel_object = RunModel(
    model_script='local_pfn.py', model_object_name='gfun_sensitivity', var_names=var_names, vec=True, a_values=a_values)

dist_object = [Uniform(), ] * na

sens = MorrisSensitivity(runmodel_object=runmodel_object,
                         distributions=dist_object,
                         n_levels=20,
                         maximize_dispersion=True)
sens.run(n_trajectories=10)


print(['a{}={}'.format(i + 1, ai) for i, ai in enumerate(a_values)])

fig, ax = plt.subplots()
ax.scatter(sens.mustar_indices, sens.sigma_indices)
for i, (mu, sig) in enumerate(zip(sens.mustar_indices, sens.sigma_indices)):
    ax.text(x=mu + 0.01, y=sig + 0.01, s='X{}'.format(i + 1))
ax.set_xlabel(r'$\mu^{\star}$', fontsize=14)
ax.set_ylabel(r'$\sigma$', fontsize=14)
ax.set_title('Morris sensitivity indices', fontsize=16)
plt.show()

shutil.rmtree(runmodel_object.model_dir)