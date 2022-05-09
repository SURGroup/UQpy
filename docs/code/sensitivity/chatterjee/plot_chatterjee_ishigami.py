r"""

Ishigami function
==============================================

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
from UQpy.sensitivity.chatterjee import Chatterjee

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
# Compute Chatterjee indices

# %%
SA = Chatterjee(runmodel_obj, dist_object)

computed_indices = SA.run(
    n_samples=100_000,
    estimate_sobol_indices=True,
    num_bootstrap_samples=100,
    confidence_level=0.95,
)

# %%
computed_indices["chatterjee_i"]

# %%
computed_indices["CI_chatterjee_i"]

# %%
computed_indices["sobol_i"]