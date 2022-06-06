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

.. [1] Fabrice Gamboa, Pierre Gremaud, Thierry Klein, and Agnès Lagnoux. (2020). Global Sensitivity Analysis: a new generation of mighty estimators based on rank statistics.

"""

# %%
import numpy as np

from UQpy.run_model.RunModel import RunModel
from UQpy.run_model.model_execution.PythonModel import PythonModel
from UQpy.distributions import Uniform
from UQpy.distributions.collection.JointIndependent import JointIndependent
from UQpy.sensitivity.Chatterjee import Chatterjee
from UQpy.sensitivity.PostProcess import *

np.random.seed(123)

# %% [markdown]
# **Define the model and input distributions**

# Create Model object
num_vars = 6
a_vals = np.array([0.0, 0.5, 3.0, 9.0, 99.0, 99.0])

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
SA = Chatterjee(runmodel_obj, dist_object)

# Compute Chatterjee indices using the pick and freeze algorithm
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
# :math:`S_1` = 5.86781190e-01
#
# :math:`S_2` = 2.60791640e-01
#
# :math:`S_3` = 3.66738244e-02
#
# :math:`S_4` = 5.86781190e-03
#
# :math:`S_5` = 5.86781190e-05
#
# :math:`S_6` = 5.86781190e-05

# %%
computed_indices["sobol_i"]

# **Plot the first order Sobol indices**
fig2, ax2 = plot_sensitivity_index(
    computed_indices["sobol_i"][:, 0],
    plot_title="First order Sobol indices",
    color="C0",
)
