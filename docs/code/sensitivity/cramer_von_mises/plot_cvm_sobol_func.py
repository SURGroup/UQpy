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
from UQpy.sensitivity.cramer_von_mises import CramervonMises as cvm

# %%
# Create Model object
num_vars = 6
a_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

model = PythonModel(
    model_script="local_sobol_func.py",
    model_object_name="evaluate",
    var_names=[r"$X_1$", "$X_2$"],
    delete_files=True,
    a_values=a_vals,
)

runmodel_obj = RunModel(model=model)

# Define distribution object
dist_object = JointIndependent([Uniform(0, 1)] * num_vars)

# %%
SA = cvm(runmodel_obj, dist_object)

# Compute Sobol indices using the pick and freeze algorithm
computed_indices = SA.run(n_samples=20_000, estimate_sobol_indices=True)

# %%
computed_indices["CVM_i"]

# %% [markdown]
# Sobol indices computed analytically
#
# $S_1$ = 0.46067666
#
# $S_2$ = 0.20474518
#
# $S_3$ = 0.11516917
#
# $S_4$ = 0.07370827
#
# $S_5$ = 0.0511863
#
# $S_6$ = 0.03760626
#

# %%
computed_indices["sobol_i"]
