r"""

Sobol function
==============================================

The Sobol function is non-linear function that is commonly used to benchmark uncertainty 
and senstivity analysis methods. Unlike the ishigami function which has 3 input 
variables, the Sobol function can have any number of input variables. 

.. math::

    g(x_1, x_2, \ldots, x_D) := \prod_{i=1}^{D} \frac{|4x_i - 2| + a_i}{1 + a_i},

where,

.. math::
    x_i \sim \mathcal{U}(0, 1), \quad a_i \in \mathbb{R}.


The function was also used in the Chatterjee indices section to demonstrate the 
computation of the Chatterjee indices. We can see clearly that the estimates are 
equivalent.

"""

# %%
import numpy as np

from UQpy.run_model.RunModel import RunModel
from UQpy.run_model.model_execution.PythonModel import PythonModel
from UQpy.distributions import Uniform
from UQpy.distributions.collection.JointIndependent import JointIndependent
from UQpy.sensitivity.CramerVonMisesSensitivity import CramerVonMisesSensitivity as cvm
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
    var_names=[r"$X_1$", "$X_2$"],
    delete_files=True,
    a_values=a_vals,
)

runmodel_obj = RunModel(model=model)

# Define distribution object
dist_object = JointIndependent([Uniform(0, 1)] * num_vars)

# %% [markdown]
# **Compute Cramér-von Mises indices**

# %%
SA = cvm(runmodel_obj, dist_object)

# Compute Sobol indices using rank statistics
SA.run(n_samples=50_000, estimate_sobol_indices=True)

# %% [markdown]
# **Cramér-von Mises indices**

# %%
SA.first_order_CramerVonMises_indices

# **Plot the CVM indices**
fig1, ax1 = plot_sensitivity_index(
    SA.first_order_CramerVonMises_indices[:, 0],
    plot_title="Cramér-von Mises indices",
    color="C4",
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
SA.total_order_sobol_indices

# **Plot the first order Sobol indices**
fig2, ax2 = plot_sensitivity_index(
    SA.total_order_sobol_indices[:, 0],
    plot_title="First order Sobol indices",
    color="C0",
)
