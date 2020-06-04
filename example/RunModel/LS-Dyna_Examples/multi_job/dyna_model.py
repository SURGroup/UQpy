from UQpy.Distributions import Uniform
from UQpy.RunModel import RunModel
from UQpy.SampleMethods import MCS

# Define the distribution objects
d1 = Uniform(loc=0.02, scale=0.06)
d2 = Uniform(loc=0.02, scale=0.01)
d3 = Uniform(loc=0.02, scale=0.01)
d4 = Uniform(loc=0.0025, scale=0.0075)
d5 = Uniform(loc=0.02, scale=0.06)
d6 = Uniform(loc=0.02, scale=0.01)
d7 = Uniform(loc=0.02, scale=0.01)
d8 = Uniform(loc=0.0025, scale=0.0075)

# Draw the samples using MCS
x = MCS(dist_object=[d1, d2, d3, d4, d5, d6, d7, d8], nsamples=12, random_state=349875)

# Run the model
run_ = RunModel(samples=x.samples, ntasks=6, model_script='dyna_script.py', input_template='dyna_input.k',
                var_names=['x0', 'y0', 'z0', 'R0', 'x1', 'y1', 'z1', 'R1'],  model_dir='dyna_test', cluster=True,
                verbose=False, fmt='{:>10.4f}', cores_per_task=12)






