"""

Cluster Script Example for Third-party
======================================
"""

# %% md
#
# In this case, we're just running a simple addition of random numbers, but
# the process is exactly the same for more complicated workflows. The pre-
# and post-processing is done through `model_script` and `output_script`
# respectively, while the computationally intensive portion of the workflow
# is launched in `cluster_script. The example below provides a minimal framework
# from which more complex cases can be constructed.
#
# Import the necessary libraries

# %%
from UQpy.sampling import LatinHypercubeSampling
from UQpy.run_model.RunModel import RunModel
from UQpy.run_model.model_execution.ThirdPartyModel import ThirdPartyModel
from UQpy.distributions import Uniform
import numpy as np
import time
import csv

# %% md
#
# Define the distribution objects.

# %%

var_names=["var_1", "var_2"]        
distributions = [Uniform(250.0, 40.0), Uniform(66.0, 24.0)]

# %% md
#
# Draw the samples using Latin Hypercube Sampling.

# %%

x_lhs = LatinHypercubeSampling(distributions, nsamples=4)

# %% md
#
# Run the model.

# %%

model = ThirdPartyModel(var_names=var_names, input_template='inputRealization.json', model_script='addition_run.py',
                        output_script='process_addition_output.py', output_object_name='OutputProcessor',
                        model_dir='AdditionRuns')

t = time.time()
modelRunner = RunModel(model=model, samples=x_lhs.samples, ntasks=1,
                       cores_per_task=2, nodes=1, resume=False,
                       run_type='CLUSTER', cluster_script='./run_script.sh')

t_total = time.time() - t
print("\nTotal time for all experiments:")
print(t_total, "\n")

# %% md
#
# Print model results--this is just for illustration

# %%
for index, experiment in enumerate(modelRunner.qoi_list, 0):
    if len(experiment.qoi) != 0:
        for item in experiment.qoi:
            print("These are the random numbers for sample {}:".format(index))
            for sample in x_lhs.samples[index]:
                print("{}\t".format(sample))

            print("This is their sum:")
            for result in item:
                print("{}\t".format(result))
        print()
