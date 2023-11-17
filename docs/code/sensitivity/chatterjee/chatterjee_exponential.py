"""

Exponential function
==============================================

The exponential function was used in [1]_ to demonstrate the 
Cramér-von Mises indices. Chattererjee indices approach the Cramér-von Mises 
indices in the sample limit and will be demonstrated via this example.

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
from UQpy.sensitivity.ChatterjeeSensitivity import ChatterjeeSensitivity
from UQpy.sensitivity.PostProcess import *

np.random.seed(123)

# %% [markdown]
# **Define the model and input distributions**

# Create Model object
model = PythonModel(
    model_script="local_exponential.py",
    model_object_name="evaluate",
    var_names=[
        "X_1",
        "X_2",
    ],
    delete_files=True,
)

runmodel_obj = RunModel(model=model)

# Define distribution object
dist_object = JointIndependent([Normal(0, 1)] * 2)

# %% [markdown]
# **Compute Chatterjee indices**

# %% [markdown]
SA = ChatterjeeSensitivity(runmodel_obj, dist_object)

# Compute Chatterjee indices using the pick and freeze algorithm
SA.run(n_samples=1_000_000)

# %% [markdown]
# **Chattererjee indices**
#
# Chattererjee indices approach the Cramér-von Mises indices in the sample limit.
#
# Expected value of the sensitivity indices:
#
# :math:`S^1_{CVM} = \frac{6}{\pi} \operatorname{arctan}(2) - 2 \approx 0.1145`
#
# :math:`S^2_{CVM} = \frac{6}{\pi} \operatorname{arctan}(\sqrt{19}) - 2 \approx 0.5693`

# %%
SA.first_order_chatterjee_indices

# **Plot the Chatterjee indices**
fig1, ax1 = plot_sensitivity_index(
    SA.first_order_chatterjee_indices[:, 0],
    plot_title="Chatterjee indices",
    color="C2",
)
