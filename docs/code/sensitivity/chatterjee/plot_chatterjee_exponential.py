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
from UQpy.sensitivity.chatterjee import Chatterjee

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
# Compute Chatterjee indices

# %%
SA = Chatterjee(runmodel_obj, dist_object)

# Compute Sobol indices using the pick and freeze algorithm
computed_indices = SA.run(n_samples=1_000_000)

# %% [markdown]
# Cramer-von Mises sensitivity analysis
#
# Expected value of the sensitivity indices:
#
# $S^1_{CVM} = \frac{6}{\pi} \operatorname{arctan}(2) - 2 \approx 0.1145$
#
# $S^2_{CVM} = \frac{6}{\pi} \operatorname{arctan}(\sqrt{19}) - 2 \approx 0.5693$

# %%
computed_indices["chatterjee_i"]