r"""

Sobol function
==============================================

.. math::

    g(x_1, x_2, \ldots, x_D) := \prod_{i=1}^{D} \frac{|4x_i - 2| + a_i}{1 + a_i},

where,

.. math::
    x_i \sim \mathcal{U}(0, 1), \quad a_i \in \mathbb{R}.

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
num_vars = 6
a_vals = np.array([0.0, 0.5, 3.0, 9.0, 99.0, 99.0])

model = PythonModel(
    model_script="local_sobol_func.py",
    model_object_name="evaluate",
    var_names=["X_" + str(i) for i in range(num_vars)],
    delete_files=True,
    a_values=a_vals,
)

runmodel_obj = RunModel(model=model)

# Define distribution object
dist_object = JointIndependent([Uniform(0, 1)] * num_vars)

# %% [markdown]
# Compute Chatterjee indices

# %%
SA = Chatterjee(runmodel_obj, dist_object)

# Compute Sobol indices using the pick and freeze algorithm
computed_indices = SA.run(n_samples=500_000, estimate_sobol_indices=True)

# %%
computed_indices["chatterjee_i"]

# %% [markdown]
# $S_1$ = 5.86781190e-01
#
# $S_2$ = 2.60791640e-01
#
# $S_3$ = 3.66738244e-02
#
# $S_4$ = 5.86781190e-03
#
# $S_5$ = 5.86781190e-05
#
# $S_6$ = 5.86781190e-05

# %%
computed_indices["sobol_i"]
