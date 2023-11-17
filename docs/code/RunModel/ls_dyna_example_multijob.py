"""

Third-party - Multi-job - LS Dyna
==================================
"""

# %% md
#
# Import the necessary libraries.

# %%
from UQpy.distributions import Uniform
from UQpy.run_model.RunModel import RunModel
from UQpy.run_model.model_types.ThirdPartyModel import ThirdPartyModel
from UQpy.sampling import MonteCarloSampling

# %% md
#
# Define the distribution objects.

# %%

d1 = Uniform(location=0.02, scale=0.06)
d2 = Uniform(location=0.02, scale=0.01)
d3 = Uniform(location=0.02, scale=0.01)
d4 = Uniform(location=0.0025, scale=0.0075)
d5 = Uniform(location=0.02, scale=0.06)
d6 = Uniform(location=0.02, scale=0.01)
d7 = Uniform(location=0.02, scale=0.01)
d8 = Uniform(location=0.0025, scale=0.0075)

# %% md
#
# Draw the samples using MCS.

# %%

x = MonteCarloSampling(distributions=[d1, d2, d3, d4, d5, d6, d7, d8], samples_number=12, random_state=349875)

# %% md
#
# Run the model.

# %%
m = ThirdPartyModel(model_script='dyna_script.py', input_template='dyna_input.k',
                    var_names=['x0', 'y0', 'z0', 'R0', 'x1', 'y1', 'z1', 'R1'], model_dir='dyna_test',
                    fmt='{:>10.4f}')
run_ = RunModel(samples=x.samples, ntasks=6, cores_per_task=12, model=m)
