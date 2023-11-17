"""

Third-party - OpenSees
==================================
"""

# %% md
#
# Import the necessary libraries.

# %%
import numpy as np

from UQpy.distributions import Uniform
from UQpy.run_model.RunModel import RunModel
from UQpy.run_model.model_types.ThirdPartyModel import ThirdPartyModel
from UQpy.sampling import MonteCarloSampling

# %% md
#
# Define the distribution objects.

# %%

dist1 = Uniform(location=15000, scale=10000)
dist2 = Uniform(location=450000, scale=80000)
dist3 = Uniform(location=2.0e8, scale=0.5e8)

# %% md
#
# Draw the samples using MCS.

# %%

x = MonteCarloSampling(distributions=[dist1, dist2, dist3] * 6, samples_number=5, random_state=938475)
samples = np.array(x.samples).round(2)

# %% md
#
# Run the model.

# %%

names_ = ['fc1', 'fy1', 'Es1', 'fc2', 'fy2', 'Es2', 'fc3', 'fy3', 'Es3', 'fc4', 'fy4', 'Es4', 'fc5', 'fy5', 'Es5',
          'fc6', 'fy6', 'Es6']

m = ThirdPartyModel(model_script='opensees_model.py', input_template='import_variables.tcl', var_names=names_,
                    model_object_name="opensees_run", output_script='process_opensees_output.py',
                    output_object_name='read_output')
opensees_rc6_model = RunModel(samples=samples, ntasks=5, model=m)

outputs = opensees_rc6_model.qoi_list
print(outputs)
