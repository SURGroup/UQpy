r"""

Ishigami function
==============================================

The ishigami function is a non-linear, non-monotonic function that is commonly used to 
benchmark uncertainty and senstivity analysis methods.

.. math::
    f(x_1, x_2, x_3) = sin(x_1) + a \cdot sin^2(x_2) + b \cdot x_3^4 sin(x_1)

.. math::
    x_1, x_2, x_3 \sim \mathcal{U}(-\pi, \pi), \quad a, b\in \mathbb{R}

"""

# %%
import numpy as np

from UQpy.run_model.RunModel import RunModel
from UQpy.run_model.model_execution.PythonModel import PythonModel
from UQpy.distributions import Uniform
from UQpy.distributions.collection.JointIndependent import JointIndependent
from UQpy.sensitivity.ChatterjeeSensitivity import ChatterjeeSensitivity
from UQpy.sensitivity.PostProcess import *

np.random.seed(123)

# %% [markdown]
# **Define the model and input distributions**

# Create Model object
model = PythonModel(
    model_script="local_ishigami.py",
    model_object_name="evaluate",
    var_names=[r"$X_1$", "$X_2$", "$X_3$"],
    delete_files=True,
    params=[7, 0.1],
)

runmodel_obj = RunModel(model=model)

# Define distribution object
dist_object = JointIndependent([Uniform(-np.pi, 2 * np.pi)] * 3)

# %% [markdown]
# **Compute Chatterjee indices**

# %% [markdown]
SA = ChatterjeeSensitivity(runmodel_obj, dist_object)

computed_indices = SA.run(
    n_samples=100_000,
    estimate_sobol_indices=True,
    n_bootstrap_samples=100,
    confidence_level=0.95,
)

# %% [markdown]
# **Chattererjee indices**

# %%
computed_indices["chatterjee_i"]

# %% [markdown]
# **Confidence intervals for the Chatterjee indices**

# %%
computed_indices["confidence_interval_chatterjee_i"]

# **Plot the Chatterjee indices**
fig1, ax1 = plot_sensitivity_index(
    computed_indices["chatterjee_i"][:, 0],
    computed_indices["confidence_interval_chatterjee_i"],
    plot_title="Chatterjee indices",
    color="C2",
)

# %% [markdown]
# **Estimated Sobol indices**
#
# Expected first order Sobol indices:
#
# :math:`S_1`: 0.3139
#
# :math:`S_2`: 0.4424
#
# :math:`S_3`: 0.0

# %%
computed_indices["sobol_i"]

# **Plot the first order Sobol indices**
fig2, ax2 = plot_sensitivity_index(
    computed_indices["sobol_i"][:, 0],
    plot_title="First order Sobol indices",
    color="C0",
)
