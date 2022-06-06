r"""

Toy multioutput function
==============================================

.. math::
    Y = f (X_{1}, X_{2}) := \left(\begin{array}{c}
                                X_{1}+X_{2}+X_{1} X_{2} \\
                                2 X_{1}+3 X_{1} X_{2}+X_{2}
                                \end{array}\right)

.. math::
    \text{case 1: } X_1, X_2 \sim \mathcal{N}(0, 1)

.. math::
    \text{case 2: } X_1, X_2 \sim \mathcal{U}(0, 1)

"""

# %%
from UQpy.run_model.RunModel import RunModel
from UQpy.run_model.model_execution.PythonModel import PythonModel
from UQpy.distributions import Uniform, Normal
from UQpy.distributions.collection.JointIndependent import JointIndependent
from UQpy.sensitivity.GeneralisedSobol import GeneralisedSobol
from UQpy.sensitivity.PostProcess import *

np.random.seed(123)

# %% [markdown]
# **Define the model and input distributions**

# Create Model object
model = PythonModel(
    model_script="local_multioutput.py",
    model_object_name="evaluate",
    var_names=[r"X_1$", r"X_2"],
    delete_files=True,
)

runmodel_obj = RunModel(model=model)

# Define distribution object
dist_object_1 = JointIndependent([Normal(0, 1)] * 2)

# %% [markdown]
# **Compute generalised Sobol indices**

# %% [markdown]
SA = GeneralisedSobol(runmodel_obj, dist_object_1)

computed_indices = SA.run(
    n_samples=20_000, confidence_level=0.95, num_bootstrap_samples=5_00
)

# %% [markdown]
# **First order Generalised Sobol indices**
#
# Expected generalised Sobol indices:
#
# Gaussian case
#
# :math:`GS_1` = 0.2941
#
# :math:`GS_2` = 0.1179

# %%
computed_indices["gen_sobol_i"]

# **Plot the first order sensitivity indices**
fig1, ax1 = plot_sensitivity_index(
    computed_indices["gen_sobol_i"][:, 0],
    confidence_interval=computed_indices["confidence_interval_gen_sobol_i"],
    plot_title="First order Generalised Sobol indices",
    color="C0",
)

# %%
computed_indices["gen_sobol_total_i"]

# **Plot the first and total order sensitivity indices**
fig2, ax2 = plot_index_comparison(
    computed_indices["gen_sobol_i"][:, 0],
    computed_indices["gen_sobol_total_i"][:, 0],
    confidence_interval_1=computed_indices["confidence_interval_gen_sobol_i"],
    confidence_interval_2=computed_indices["confidence_interval_gen_sobol_total_i"],
    label_1="First order",
    label_2="Total order",
    plot_title="First and Total order Generalised Sobol indices",
)

# %% [markdown]
# **Compute generalised Sobol indices**

# %% [markdown]
dist_object_2 = JointIndependent([Uniform(0, 1)] * 2)

SA = GeneralisedSobol(runmodel_obj, dist_object_2)

computed_indices = SA.run(
    n_samples=20_000, confidence_level=0.95, num_bootstrap_samples=5_00
)

# %% [markdown]
# **First order Generalised Sobol indices**
#
# Expected generalised Sobol indices:
#
# Uniform case
#
# :math:`GS_1` = 0.6084
#
# :math:`GS_2` = 0.3566

# %%
computed_indices["gen_sobol_i"]

# **Plot the first order sensitivity indices**
fig3, ax3 = plot_sensitivity_index(
    computed_indices["gen_sobol_i"][:, 0],
    confidence_interval=computed_indices["confidence_interval_gen_sobol_i"],
    plot_title="First order Generalised Sobol indices",
    color="C0",
)

# %%
computed_indices["gen_sobol_total_i"]

# **Plot the first and total order sensitivity indices**
fig4, ax4 = plot_index_comparison(
    computed_indices["gen_sobol_i"][:, 0],
    computed_indices["gen_sobol_total_i"][:, 0],
    confidence_interval_1=computed_indices["confidence_interval_gen_sobol_i"],
    confidence_interval_2=computed_indices["confidence_interval_gen_sobol_total_i"],
    label_1="First order",
    label_2="Total order",
    plot_title="First and Total order Generalised Sobol indices",
)
