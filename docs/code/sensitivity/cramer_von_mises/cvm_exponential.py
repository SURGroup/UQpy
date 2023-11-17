"""

Exponential function
==============================================

The exponential function was used in [1]_ to demonstrate the 
Cramér-von Mises indices.

.. math::
    f(x) := \exp(x_1 + 2x_2), \quad x_1, x_2 \sim \mathcal{N}(0, 1)

.. [1] Gamboa, F., Klein, T., & Lagnoux, A. (2018). Sensitivity Analysis Based on \
Cramér-von Mises Distance. SIAM/ASA Journal on Uncertainty Quantification, 6(2), \
522-548. doi:10.1137/15M1025621. (`Link <https://doi.org/10.1137/15M1025621>`_)

"""

# %%
from UQpy.run_model.RunModel import RunModel
from UQpy.run_model.model_types.PythonModel import PythonModel
from UQpy.distributions import Normal
from UQpy.distributions.collection.JointIndependent import JointIndependent
from UQpy.sensitivity.CramerVonMisesSensitivity import CramerVonMisesSensitivity as cvm
from UQpy.sensitivity.PostProcess import *

np.random.seed(123)

# %% [markdown]
# **Define the model and input distributions**

# Create Model object
model = PythonModel(
    model_script="local_exponential.py",
    model_object_name="evaluate",
    var_names=[r"$X_1$", "$X_2$"],
    delete_files=True,
)

runmodel_obj = RunModel(model=model)

# Define distribution object
dist_object = JointIndependent([Normal(0, 1)] * 2)

# %% [markdown]
# **Compute Cramér-von Mises indices**

# %%
SA = cvm(runmodel_obj, dist_object)

# Compute CVM indices using the pick and freeze algorithm
SA.run(n_samples=20_000, estimate_sobol_indices=True)

# %% [markdown]
# **Cramér-von Mises indices**
#
# Expected value of the sensitivity indices:
#
# :math:`S^1_{CVM} = \frac{6}{\pi} \operatorname{arctan}(2) - 2 \approx 0.1145`
#
# :math:`S^2_{CVM} = \frac{6}{\pi} \operatorname{arctan}(\sqrt{19}) - 2 \approx 0.5693`

# %%
SA.first_order_CramerVonMises_indices

# **Plot the CVM indices**
fig1, ax1 = plot_sensitivity_index(
    SA.first_order_CramerVonMises_indices[:, 0],
    plot_title="Cramér-von Mises indices",
    color="C4",
)

# %% [markdown]
# **Estimated first order Sobol indices**
#
# Expected first order Sobol indices:
#
# :math:`S_1` = 0.0118
#
# :math:`S_2` = 0.3738

# %%
SA.first_order_sobol_indices

# **Plot the first order Sobol indices**
fig2, ax2 = plot_sensitivity_index(
    SA.first_order_sobol_indices[:, 0],
    plot_title="First order Sobol indices",
    color="C0",
)

# %% [markdown]
# **Estimated total order Sobol indices**

# %%
SA.total_order_sobol_indices

# **Plot the first and total order sensitivity indices**
fig3, ax3 = plot_index_comparison(
    SA.first_order_sobol_indices[:, 0],
    SA.total_order_sobol_indices[:, 0],
    label_1="First order Sobol indices",
    label_2="Total order Sobol indices",
    plot_title="First and Total order Sobol indices",
)
