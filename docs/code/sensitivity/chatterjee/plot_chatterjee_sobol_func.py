r"""

Sobol function
==============================================

The Sobol function is non-linear function that is commonly used to benchmark uncertainty 
and senstivity analysis methods. Unlike the ishigami function which has 3 input 
variables, the Sobol function can have any number of input variables. 

This function was used in [1]_ to compare the Pick and Freeze approach and the rank 
statistics approach to estimating Sobol indices. The rank statistics approach was 
observed to be more accurate than the Pick and Freeze approach and it also provides 
better estimates when only a small number of model evaluations are available.

.. math::

    g(x_1, x_2, \ldots, x_D) := \prod_{i=1}^{D} \frac{|4x_i - 2| + a_i}{1 + a_i},

where,

.. math::
    x_i \sim \mathcal{U}(0, 1), \quad a_i \in \mathbb{R}.

Finally, we also compare the convergence rate of the Pick and Freeze approach with the
rank statistics approach as in [1]_.

.. [1] Fabrice Gamboa, Pierre Gremaud, Thierry Klein, and Agnès Lagnoux. (2020). Global Sensitivity Analysis: a new generation of mighty estimators based on rank statistics. (`Link <https://arxiv.org/abs/2003.01772>`_)

"""

# %%
import numpy as np

from UQpy.run_model.RunModel import RunModel
from UQpy.run_model.model_execution.PythonModel import PythonModel
from UQpy.distributions import Uniform
from UQpy.distributions.collection.JointIndependent import JointIndependent
from UQpy.sensitivity.ChatterjeeSensitivity import ChatterjeeSensitivity
from UQpy.sensitivity.SobolSensitivity import SobolSensitivity
from UQpy.sensitivity.PostProcess import *

np.random.seed(123)

# %% [markdown]
# **Define the model and input distributions**

# Create Model object
num_vars = 6
a_vals = np.arange(1, num_vars+1, 1)

model = PythonModel(
    model_script="local_sobol_func.py",
    model_object_name="evaluate",
    var_names=["X_" + str(i) for i in range(num_vars)],
    delete_files=True,
    a_values=a_vals,
)

runmodel_obj = RunModel(model=model)

# Define distribution object
dist_object = JointIndependent([Uniform(0, 1)] * num_vars)

# %% [markdown]
# **Compute Chatterjee indices**

# %% [markdown]
SA = ChatterjeeSensitivity(runmodel_obj, dist_object)

# Compute Chatterjee indices using rank statistics
computed_indices = SA.run(n_samples=500_000, estimate_sobol_indices=True)

# %% [markdown]
# **Chatterjee indices**

# %%
computed_indices["chatterjee_i"]

# **Plot the Chatterjee indices**
fig1, ax1 = plot_sensitivity_index(
    computed_indices["chatterjee_i"][:, 0],
    plot_title="Chatterjee indices",
    color="C2",
)

# %% [markdown]
# **Estimated Sobol indices**
#
# Expected first order Sobol indices:
#
# :math:`S_1` = 0.46067666
#
# :math:`S_2` = 0.20474518
#
# :math:`S_3` = 0.11516917
#
# :math:`S_4` = 0.07370827
#
# :math:`S_5` = 0.0511863
#
# :math:`S_6` = 0.03760626

# %%
computed_indices["sobol_i"]

# **Plot the first order Sobol indices**
fig2, ax2 = plot_sensitivity_index(
    computed_indices["sobol_i"][:, 0],
    plot_title="First order Sobol indices",
    color="C0",
)

# %% [markdown]
# **Comparing convergence rate of rank statistics and the Pick and Freeze approach**
#
# In the Pick-Freeze estimations, several sizes of sample N have been considered: 
# N = 100, 500, 1000, 5000, 10000, 50000, and 100000. 
# The Pick-Freeze procedure requires (p + 1) samples of size N. 
# To have a fair comparison, the sample sizes considered in the estimation using 
# rank statistics are n = (p+1)N = 7N. 
# We observe that both methods converge and give precise results for large sample sizes.

# %%

# Compute indices values for equal number of model evaluations

true_values = np.array([0.46067666, 
                        0.20474518, 
                        0.11516917, 
                        0.07370827, 
                        0.0511863 ,
                        0.03760626])

sample_sizes = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]
num_studies = len(sample_sizes)

store_pick_freeze = np.zeros((num_vars, num_studies))
store_rank_stats = np.zeros((num_vars, num_studies))

SA_chatterjee = ChatterjeeSensitivity(runmodel_obj, dist_object)
SA_sobol = SobolSensitivity(runmodel_obj, dist_object)

for i, sample_size in enumerate(sample_sizes):

    # Estimate using rank statistics
    _indices = SA_chatterjee.run(n_samples=sample_size*7, estimate_sobol_indices=True)
    store_rank_stats[:, i] = _indices["sobol_i"].ravel()

    # Estimate using Pick and Freeze approach
    _indices = SA_sobol.run(n_samples=sample_size)
    store_pick_freeze[:, i] = _indices["sobol_i"].ravel()

# %%

## Convergence plot

fix, ax = plt.subplots(2, 3, figsize=(30, 15))

for k in range(num_vars):

    i, j = divmod(k, 3) # (built-in) divmod(a, b) returns a tuple (a // b, a % b)

    ax[i][j].semilogx(sample_sizes, store_rank_stats[k, :], 'ro-', label='Chatterjee estimate')
    ax[i][j].semilogx(sample_sizes, store_pick_freeze[k, :], 'bx-', label='Pick and Freeze estimate')
    ax[i][j].hlines(true_values[k], 0, sample_sizes[-1], 'k', label='True indices')
    ax[i][j].set_title(r'$S^' + str(k+1) + '$ = ' + str(np.round(true_values[k], 4)))

plt.suptitle('Comparing convergence of the Chatterjee estimate and the Pick and Freeze approach')
plt.legend()
plt.show()
