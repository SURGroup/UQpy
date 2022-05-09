r"""

Toy multioutput function
==============================================

.. math::
    Y = f (X_{1}, X_{2}) := \left(\begin{array}{c}
                                X_{1}+X_{2}+X_{1} X_{2} \\
                                2 X_{1}+3 X_{1} X_{2}+X_{2}
                                \end{array}\right)

.. math::
    \text{case 1: } X_1, X_2 \sim \mathcal{U}(0, 1)

.. math::
    \text{case 2: } X_1, X_2 \sim \mathcal{N}(0, 1)

"""

# %%
from UQpy.run_model.RunModel import RunModel
from UQpy.run_model.model_execution.PythonModel import PythonModel
from UQpy.distributions import Uniform, Normal
from UQpy.distributions.collection.JointIndependent import JointIndependent
from UQpy.sensitivity.generalised_sobol import GeneralisedSobol

# %%
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
dist_object_2 = JointIndependent([Uniform(0, 1)] * 2)

# %%
SA = GeneralisedSobol(runmodel_obj, dist_object_1)

computed_indices = SA.run(
    n_samples=20_000, confidence_level=0.95, num_bootstrap_samples=5_00
)

# %% [markdown]
# Gaussian case
#
# $S_1$ = 0.2941
#
# $S_2$ = 0.1179

# %%
computed_indices["gen_sobol_i"]

# %%
computed_indices["gen_sobol_total_i"]

# %%
SA = GeneralisedSobol(runmodel_obj, dist_object_2)

computed_indices = SA.run(n_samples=100_000)

# %% [markdown]
# Gaussian case
#
# $S_1$ = 0.6084
#
# $S_2$ = 0.3566

# %%
computed_indices["gen_sobol_i"]

# %%
computed_indices["gen_sobol_total_i"]
