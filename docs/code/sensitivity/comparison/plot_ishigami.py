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
from UQpy.sensitivity.CramerVonMisesSensitivity import CramerVonMisesSensitivity as cvm
from UQpy.sensitivity.SobolSensitivity import SobolSensitivity
from UQpy.sensitivity.PostProcess import *

np.random.seed(123)

# %% [markdown]
# **Define the model and input distributions**

# %%
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
# **Compute Sobol indices**

# %%
SA_sobol = SobolSensitivity(runmodel_obj, dist_object)

computed_indices_sobol = SA_sobol.run(n_samples=100_000)

# %% [markdown]
# **First order Sobol indices**
#
# Expected first order Sobol indices:
#
# :math:`S_1` = 0.3139
#
# :math:`S_2` = 0.4424
#
# :math:`S_3` = 0.0

# %%
computed_indices_sobol["sobol_i"]

# %% [markdown]
# **Total order Sobol indices**
#
# Expected total order Sobol indices:
#
# :math:`S_{T_1}` = 0.55758886
#
# :math:`S_{T_2}` = 0.44241114
#
# :math:`S_{T_3}` =  0.24368366

# %%
computed_indices_sobol["sobol_total_i"]

# %% [markdown]
# **Compute Chatterjee indices**

# %% [markdown]
SA_chatterjee = ChatterjeeSensitivity(runmodel_obj, dist_object)

computed_indices_chatterjee = SA_chatterjee.run(n_samples=50_000)

# %%
computed_indices_chatterjee["chatterjee_i"]

# %% [markdown]
# **Compute Cramér-von Mises indices**
SA_cvm = cvm(runmodel_obj, dist_object)

# Compute CVM indices using the pick and freeze algorithm
computed_indices_cvm = SA_cvm.run(n_samples=20_000, estimate_sobol_indices=True)

# %%
computed_indices_cvm["CVM_i"]

# %%
# **Plot all indices**

num_vars = 3
_idx = np.arange(num_vars)
variable_names = [r"$X_{}$".format(i + 1) for i in range(num_vars)]

# round to 2 decimal places
indices_1 = np.around(computed_indices_sobol["sobol_i"][:, 0], decimals=2)
indices_2 = np.around(computed_indices_chatterjee["chatterjee_i"][:, 0], decimals=2)
indices_3 = np.around(computed_indices_cvm["CVM_i"][:, 0], decimals=2)

fig, ax = plt.subplots()
width = 0.3
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

bar_indices_1 = ax.bar(
    _idx - width,  # x-axis
    indices_1,  # y-axis
    width=width,  # bar width
    color="C0",  # bar color
    # alpha=0.5,  # bar transparency
    label="Sobol",  # bar label
    ecolor="k",  # error bar color
    capsize=5,  # error bar cap size in pt
)

bar_indices_2 = ax.bar(
    _idx,  # x-axis
    indices_2,  # y-axis
    width=width,  # bar width
    color="C2",  # bar color
    # alpha=0.5,  # bar transparency
    label="Chatterjee",  # bar label
    ecolor="k",  # error bar color
    capsize=5,  # error bar cap size in pt
)

bar_indices_3 = ax.bar(
    _idx + width,  # x-axis
    indices_3,  # y-axis
    width=width,  # bar width
    color="C3",  # bar color
    # alpha=0.5,  # bar transparency
    label="Cramér-von Mises",  # bar label
    ecolor="k",  # error bar color
    capsize=5,  # error bar cap size in pt
)

ax.bar_label(bar_indices_1, label_type="edge", fontsize=10)
ax.bar_label(bar_indices_2, label_type="edge", fontsize=10)
ax.bar_label(bar_indices_3, label_type="edge", fontsize=10)
ax.set_xticks(_idx, variable_names)
ax.set_xlabel("Model inputs")
ax.set_title("Comparison of sensitivity indices")
ax.set_ylim(top=1)  # set only upper limit of y to 1
ax.legend()

plt.show()
