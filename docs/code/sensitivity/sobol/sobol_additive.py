"""

Additive function
==============================================

We introduce the variance-based Sobol indices using an elementary example. 
For more details, refer [1]_.

.. math::
    f(x) = a \cdot X_1 + b \cdot X_2, \quad X_1, X_2 \sim \mathcal{N}(0, 1), \quad a,b \in \mathbb{R}

.. [1] Saltelli A, T. (2008). Global sensitivity analysis: The primer. John Wiley.

"""

# %%

from UQpy.run_model.RunModel import RunModel
from UQpy.run_model.model_types.PythonModel import PythonModel
from UQpy.distributions import Normal
from UQpy.distributions.collection.JointIndependent import JointIndependent
from UQpy.sensitivity.SobolSensitivity import SobolSensitivity
from UQpy.sensitivity.PostProcess import *

np.random.seed(123)

# %% [markdown]
# **Define the model and input distributions**

# Create Model object
a, b = 1, 2

model = PythonModel(
    model_script="local_additive.py",
    model_object_name="evaluate",
    var_names=[
        "X_1",
        "X_2",
    ],
    delete_files=True,
    params=[a, b],
)

runmodel_obj = RunModel(model=model)

# Define distribution object
dist_object = JointIndependent([Normal(0, 1)] * 2)

# %% [markdown]
# **Compute Sobol indices**

# %% [markdown]
SA = SobolSensitivity(runmodel_obj, dist_object)

SA.run(n_samples=50_000)

# %% [markdown]
# **First order Sobol indices**
#
# Expected first order Sobol indices:
#
# :math:`\mathrm{S}_1 = \frac{a^2 \cdot \mathbb{V}[X_1]}{a^2 \cdot \mathbb{V}[X_1] + b^2 \cdot \mathbb{V}[X_2]} = \frac{1^2 \cdot 1}{1^2 \cdot 1 + 2^2 \cdot 1} = 0.2`
#
# :math:`\mathrm{S}_2 = \frac{b^2 \cdot \mathbb{V}[X_2]}{a^2 \cdot \mathbb{V}[X_1] + b^2 \cdot \mathbb{V}[X_2]} = \frac{2^2 \cdot 1}{1^2 \cdot 1 + 2^2 \cdot 1} = 0.8`

# %%
SA.first_order_indices

# %%
# **Plot the first and total order sensitivity indices**
fig1, ax1 = plot_index_comparison(
    SA.first_order_indices[:, 0],
    SA.total_order_indices[:, 0],
    label_1="First order Sobol indices",
    label_2="Total order Sobol indices",
    plot_title="First and Total order Sobol indices",
)
