"""

Exponential function
==============================================

.. math::
    f(x) := \exp(x_1 + 2x_2), \quad x_1, x_2 \sim \mathcal{N}(0, 1)

"""

# %%
from UQpy.run_model.RunModel import RunModel
from UQpy.run_model.model_execution.PythonModel import PythonModel
from UQpy.distributions import Normal
from UQpy.distributions.collection.JointIndependent import JointIndependent
from UQpy.sensitivity.sobol import Sobol

# %%
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
# Compute Sobol indices

# %%
SA = Sobol(runmodel_obj, dist_object)

# Compute Sobol indices using the pick and freeze algorithm
computed_indices = SA.run(
    n_samples=100_000, num_bootstrap_samples=1_000, confidence_level=0.95
)

# %% [markdown]
# Expected first order Sobol indices (computed analytically):
#
# X1: 0.0118
#
# X2: 0.3738

# %%
computed_indices["sobol_i"]

# %% [markdown]
# Confidence intervals for first order Sobol indices

# %%
computed_indices["CI_sobol_i"]
