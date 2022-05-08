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
from UQpy.sensitivity.sobol import Sobol

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
# #### Compute Sobol indices

# %%
SA = Sobol(runmodel_obj, dist_object)

# Compute Sobol indices using the pick and freeze algorithm
computed_indices = SA.run(n_samples=50_000, estimate_second_order=True)

# %% [markdown]
# First order Sobol indices
#
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

# %% [markdown]
# Total order Sobol indices
#
# $S_{T_1}$ = 6.90085892e-01
#
# $S_{T_2}$ = 3.56173364e-01
#
# $S_{T_3}$ = 5.63335422e-02
#
# $S_{T_4}$ = 9.17057664e-03
#
# $S_{T_5}$ = 9.20083854e-05
#
# $S_{T_6}$ = 9.20083854e-05
#

# %%
computed_indices["sobol_total_i"]

# %% [markdown]
# Second-order Sobol indices
#
# $S_{12}$ = 0.0869305
#
# $S_{13}$ = 0.0122246
#
# $S_{14}$ = 0.00195594
#
# $S_{15}$ = 0.00001956
#
# $S_{16}$ = 0.00001956
#
# $S_{23}$ = 0.00543316
#
# $S_{24}$ = 0.00086931
#
# $S_{25}$ = 0.00000869
#
# $S_{26}$ = 0.00000869
#
# $S_{34}$ = 0.00012225
#
# $S_{35}$ = 0.00000122
#
# $S_{36}$ = 0.00000122
#
# $S_{45}$ = 0.00000020
#
# $S_{46}$ = 0.00000020
#
# $S_{56}$ = 2.0e-9

# %%
computed_indices["sobol_ij"]
