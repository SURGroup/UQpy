r"""

Sobol function
==============================================

The Sobol function is non-linear function that is commonly used to benchmark uncertainty 
and senstivity analysis methods. Unlike the Ishigami function which has 3 input 
variables, the Sobol function can have any number of input variables. 

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
from UQpy.sensitivity.Sobol import Sobol

# %% [markdown]
# **Define the model and input distributions**

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
# **Compute Sobol indices**

# %% [markdown]
SA = Sobol(runmodel_obj, dist_object)

# Compute Sobol indices using the pick and freeze algorithm
computed_indices = SA.run(n_samples=50_000, estimate_second_order=True)

# %% [markdown]
# **First order Sobol indices**
#
# Expected first order Sobol indices:
#
# :math:`S_1` = 5.86781190e-01
#
# :math:`S_2` = 2.60791640e-01
#
# :math:`S_3` = 3.66738244e-02
#
# :math:`S_4` = 5.86781190e-03
#
# :math:`S_5` = 5.86781190e-05
#
# :math:`S_6` = 5.86781190e-05

# %%
computed_indices["sobol_i"]

# %% [markdown]
# **Total order Sobol indices**
#
# Expected total order Sobol indices:
#
# :math:`S_{T_1}` = 6.90085892e-01
#
# :math:`S_{T_2}` = 3.56173364e-01
#
# :math:`S_{T_3}` = 5.63335422e-02
#
# :math:`S_{T_4}` = 9.17057664e-03
#
# :math:`S_{T_5}` = 9.20083854e-05
#
# :math:`S_{T_6}` = 9.20083854e-05
#

# %%
computed_indices["sobol_total_i"]

# %% [markdown]
# **Second order Sobol indices**
#
# Expected second order Sobol indices:
#
# :math:`S_{T_{12}}` = 0.0869305
#
# :math:`S_{T_{13}}` = 0.0122246
#
# :math:`S_{T_{14}}` = 0.00195594
#
# :math:`S_{T_{15}}` = 0.00001956
#
# :math:`S_{T_{16}}` = 0.00001956
#
# :math:`S_{T_{23}}` = 0.00543316
#
# :math:`S_{T_{24}}` = 0.00086931
#
# :math:`S_{T_{25}}` = 0.00000869
#
# :math:`S_{T_{26}}` = 0.00000869
#
# :math:`S_{T_{34}}` = 0.00012225
#
# :math:`S_{T_{35}}` = 0.00000122
#
# :math:`S_{T_{36}}` = 0.00000122
#
# :math:`S_{T_{45}}` = 0.00000020
#
# :math:`S_{T_{46}}` = 0.00000020
#
# :math:`S_{T_{56}}` = 2.0e-9

# %%
computed_indices["sobol_ij"]
