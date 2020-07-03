import pickle
import time

from UQpy.Distributions import Normal, Uniform
from UQpy.RunModel import *
from UQpy.SampleMethods import MCS

calling_directory = os.getcwd()
t = time.time()
# Building the model
# There are two probabilistic input variables, the fire load density and the yield strength.
var_names = ['qtd', 'fy']

# Create the model object
abaqus_sfe_model = RunModel(model_script='abaqus_fire_analysis.py', input_template='abaqus_input.py',
                            output_script='extract_abaqus_output.py', var_names=var_names, ntasks=24,
                            model_dir='SFE_MCS', verbose=True, cores_per_task=1)
print('Example: Created the model object.')

# Towards defining the sampling scheme
# The fire load density is assumed to be uniformly distributed between 50 MJ/number_of_variables^2 and 450 MJ/number_of_variables^2.
# The yield strength is assumed to be normally distributed, with the parameters
# being: mean = 250 MPa and coefficient of variation of 7%.

# Creating samples using MCS
d_n = Normal(loc=50, scale=400)
d_u = Uniform(loc=2.50e8, scale=1.75e7)
x_mcs = MCS(dist_object=[d_n, d_u], nsamples=100, random_state=987979)

# Running simulations using the previously defined model object and samples
sample_points = x_mcs.samples
abaqus_sfe_model.run(samples=sample_points)

# The outputs from the analysis are the values of the performance function.
qois = abaqus_sfe_model.qoi_list

# Save the samples and the qois in a dictionary called results with keys 'inputs' and 'outputs'
results = {'inputs': sample_points, 'outputs': qois}

# Pickle the results dictionary in the current directory. The basename and extension of the desired pickle file:
res_basename = 'MCS_results'
res_extension = '.pkl'

# Create a new results file with a larger index than any existing results files with the same name in the current
# directory.
res_file_list = glob.glob(res_basename + '_???' + res_extension)
if len(res_file_list) == 0:
    res_file_name = res_basename + '_000' + res_extension
else:
    max_number = max(res_file_list).split('.')[0].split('_')[-1]
    res_file_name = res_basename + '_%03d' % (int(max_number) + 1) + res_extension

res_file_name = os.path.join(calling_directory, res_file_name)
# Save the results to this new file.
with open(res_file_name, 'wb') as f:
    pickle.dump(results, f)
print('Saved the results to ' + res_file_name)

print('Example: Done!')
print('Time elapsed: %.2f minutes' % float((time.time() - t) / 60.0))
