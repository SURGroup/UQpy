"""

Third-party - Matlab
==================================
"""


# %% md
#
# The RunModel class is capable of passing input in different formats into a single computational model. This means that
# the samples passed into a model can be passed as:
# - floating point values
# - numpy arrays
# - lists
# - tuples
# - lists of other iterables
# - numpy arrays of other iterables
# - or any combination of the above
#
# In the examples below, we demonstrate the use of a third-party computational model (in this case, Matlab) with inputs
# that are combinations of the above.
#
# Some notes on their use:
# 1. UQpy converts all sample input to a numpy array with at least two dimensions. The first dimension,
# i.e. len(samples) must correspond to the number of samples being passed for model execution. The second dimension,
# i.e. len(samples[0]) must correspond to the number of variables that each sample possesses.
# 2. Each individual sample, i.e. sample[j], may be composed of multiple data types -- with each variable having a
# different data type. For example, sample[j][k] may be a floating point value and sample[j][l] may be an array of
# arbitrary dimension.
# 3. If a specific variable has multiple dimensions, the user may specify the index to be return in the input file.
# For example, the place holder for a variable x1 corresponding to sample[j][l] that is an array of shape (1,4) can be
# read as <x1[0, 3]>, which will return the final (0,3) component of samples[j][l].
# 4. If the user does not specify the index for a multidimensional variable, then the entire multidimensional variable
# is flattened and written with comma delimiters.
#
# All examples are run using Matlab execution through the command line. The user will need to modify the model_script
# in order to provide the correct path to the Matlab application on his/her computer.

# %%

# %% md
#
# Examples 1-2:
# The provided Matlab models take the sum of three random variables:
# .. math:: s = \sum_{i=1}^3 x_i
#
# .. math:: x_i \sim N(0,1)
#
# Example 3:
# The provided Matlab model takes the product of a random variable and the determinant of a random matrix:
#
# .. math::z = x \det(Y)
#
# .. math:: x \sim N(0,1)
# :math:`y` is a :math:`3x3` matrix of standard normal random variables.

# %%

from UQpy.sampling import MonteCarloSampling
from UQpy.RunModel import RunModel
from UQpy.distributions import Normal
import matplotlib.pyplot as plt
import time
import numpy as np

# %% md
#
# Pick which model to run
# -----------------------
#
# Options:
# - 'all'
# - 'scalar'
# - 'vector'
# - 'mixed'

# %%

pick_model = 'all'

# %% md
#
# Example 1: Three scalar random variables
# ------------------------------------------
# In this example, we pass three scalar random variables. Note that this is different from assigning a single variable
# with three components, which will be handled in the following example.
#
# Here we will pass the samples both as an ndarray and as a list. Recall that UQpy converts all samples into an ndarray
# of at least two dimensions internally.

# %%

if pick_model == 'scalar' or pick_model == 'vector' or pick_model == 'all':
    # Call MCS to generate samples
    d = Normal(loc=0, scale=1)
    x_mcs = MonteCarloSampling(distributions=[d, d, d], nsamples=5, random_state=987979)
    names = ['var1', 'var11', 'var111']

    # UQpy returns samples as an ndarray. Convert them to a list for part 1.2
    x_mcs_list = list(x_mcs.samples)
    print("Monte Carlo samples of three random variables from a standard normal distribution.")
    print('Samples stored as an array:')
    print('Data type:', type(x_mcs.samples))
    print('Number of samples:', len(x_mcs.samples))
    print('Dimensions of samples:', np.shape(x_mcs.samples))
    print('Samples')
    print(x_mcs.samples)
    print()
    print('Samples stored as a list:')
    print('Data type:', type(x_mcs_list))
    print('Number of samples:', len(x_mcs_list))
    print('Dimensions of samples:', np.shape(x_mcs_list))
    print('Samples:')
    print(x_mcs_list)

# %% md
#
# 1.1 Pass sampled as ndarray, specify format in generated input file, serial execution
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This examples uses the following files:
# - model_script = matlab_model_sum_scalar.py
# - input_template = sum_scalar.m
# - output_script = process_matlab_output.py

# %%

if pick_model == 'scalar' or pick_model == 'all':
    # Call to RunModel - Here we run the model while instantiating the RunModel object.
    t = time.time()
    m = RunModel(ntasks=1, model_script='matlab_model_sum_scalar.py',
                 input_template='sum_scalar.m', var_names=names, model_object_name="matlab",
                 output_script='process_matlab_output.py', output_object_name='read_output',
                 resume=False, model_dir='Matlab_Model', fmt="{:>10.4f}", verbose=True)
    m.run(x_mcs.samples)
    t_ser_matlab = time.time() - t
    print("\nTime for serial execution:")
    print(t_ser_matlab)
    print()
    print("The values returned from the Matlab simulation:")
    print(m.qoi_list)


# %% md
#
# 1.2 Samples passed as list, no format specification, parallel execution
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This examples uses the following files:
# - model_script = matlab_model_sum_scalar.py
# - input_template = sum_scalar.m
# - output_script = process_matlab_output.py

# %%

if pick_model == 'scalar' or pick_model == 'all':
    # Call to RunModel with samples as a list - Again we run the model while instantiating the RunModel object.
    t = time.time()
    m = RunModel(samples=x_mcs_list, ntasks=2, model_script='matlab_model_sum_scalar.py',
                 input_template='sum_scalar.m', var_names=names, model_object_name="matlab",
                 output_script='process_matlab_output.py', output_object_name='read_output', resume=False,
                 model_dir='Matlab_Model', verbose=True)
    t_par_matlab = time.time() - t
    print("\nTime for parallel execution:")
    print(t_par_matlab)
    print()
    print("The values retured from the Matlab simulation:")
    print(m.qoi_list)


# %% md
#
# Example 2: Single tri-variate random variable
# -----------------------------------------------
# In this example, we pass three random variables in as a trivariate random variable. Note that this is different from
# assigning three scalar random variables, which was be handled in Example 1.
#
# Again, we will pass the samples both as an ndarray and as a list. Recall that UQpy converts all samples into an
# ndarray of at least two dimensions internally.

# %%

# %% md
#
# Restructure the samples
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# To pass the samples in as a single tri-variate variable, we need reshape the samples from shape (5, 3) to
# shape (5, 1, 3)

# %%

if pick_model == 'vector' or pick_model == 'all':
    x_mcs_tri = x_mcs.samples.reshape(5, 1, 3)
    x_mcs_tri_list = list(x_mcs_tri)

    print("Monte Carlo samples of three random variables from a standard normal distribution.")
    print('Samples stored as an array:')
    print('Data type:', type(x_mcs_tri))
    print('Number of samples:', len(x_mcs_tri))
    print('Dimensions of samples:', np.shape(x_mcs_tri))
    print('Samples')
    print(x_mcs_tri)
    print()
    print('Samples stored as a list:')
    print('Data type:', type(x_mcs_tri_list))
    print('Number of samples:', len(x_mcs_tri_list))
    print('Dimensions of samples:', np.shape(x_mcs_tri_list))
    print('Samples:')
    print(x_mcs_tri_list)

# %% md
#
# 2.1 Pass samples as ndarray, specify format in generated input file, serial execution, index samples in input_template
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This examples uses the following files:
# - model_script = matlab_model_sum_vector_indexed.py
# - input_template = sum_vector_indexed.m
# - output_script = process_matlab_output.py

# %%


if pick_model == 'vector' or pick_model == 'all':
    # Call to RunModel - Here we run the model while instantiating the RunModel object.
    # Notice that we do not specify var_names. This will default to a single variable with name x0. In this case,
    # we will read them in by indexing in the input_template.
    t = time.time()
    m = RunModel(samples=x_mcs_tri, ntasks=1, model_script='matlab_model_sum_vector_indexed.py',
                 input_template='sum_vector_indexed.m', model_object_name="matlab",
                 output_script='process_matlab_output.py', output_object_name='read_output',
                 resume=False, model_dir='Matlab_Model', fmt="{:>10.4f}")
    t_ser_matlab = time.time() - t
    print("\nTime for serial execution:")
    print(t_ser_matlab)
    print()
    print("The values returned from the Matlab simulation:")
    print(m.qoi_list)

# %% md
#
# 2.2 Samples passed as list, no format specification, parallel execution, index samples
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This examples uses the following files:
# - model_script = matlab_model_sum_vector_indexed.py
# - input_template = sum_vector_indexed.m
# - output_script = process_matlab_output.py

# %%

if pick_model == 'vector' or pick_model == 'all':
    # Call to RunModel - Here we run the model while instantiating the RunModel object.
    # Notice that we do not specify var_names. This will default to a single variable with name x0. In this case,
    # we will read them in by indexing in the input_template.
    t = time.time()
    m = RunModel(samples=x_mcs_tri_list, ntasks=2, model_script='matlab_model_sum_vector_indexed.py',
                 input_template='sum_vector_indexed.m', model_object_name="matlab",
                 output_script='process_matlab_output.py', output_object_name='read_output',
                 resume=False, model_dir='Matlab_Model')
    t_ser_matlab = time.time() - t
    print("\nTime for parallel execution:")
    print(t_ser_matlab)
    print()
    print("The values returned from the Matlab simulation:")
    print(m.qoi_list)

# %% md
#
# 2.3 Samples passed as a ndarray, specify format in generated input file, serial execution, do not index samples
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This examples uses the following files:
# - model_script = matlab_model_sum_vector_indexed.py
# - input_template = sum_vector_indexed.m
# - output_script = process_matlab_output.py

# %%

if pick_model == 'vector' or pick_model == 'all':
    # Call to RunModel - Here we run the model while instantiating the RunModel object.
    # Notice that we do not specify var_names. This will default to a single variable with name x0. In this case,
    # we will read them in by indexing in the input_template.
    t = time.time()
    m = RunModel(samples=x_mcs_tri, ntasks=1, model_script='matlab_model_sum_vector.py',
                 input_template='sum_vector.m', model_object_name="matlab",
                 output_script='process_matlab_output.py', output_object_name='read_output',
                 resume=False, model_dir='Matlab_Model', fmt="{:>10.4f}")
    t_ser_matlab = time.time() - t
    print("\nTime for serial execution:")
    print(t_ser_matlab)
    print()
    print("The values returned from the Matlab simulation:")
    print(m.qoi_list)


# %% md
#
# Example 3: Passing a scalar and an array to RunModel
# -----------------------------------------------------
# In this example, we pass a single scalar random variable as well as an array into a Matlab model.
#
# Again, we will pass the samples both as an ndarray and as a list. Recall that UQpy converts all samples into an
# ndarray of at least two dimensions internally.

# %%

# %% md
#
# Create the input samples to be passed

# %%

if pick_model == 'mixed' or pick_model == 'vector' or pick_model == 'all':

    # Call MCS to generate samples
    # First generate the scalar random variable
    x_mcs1 = MonteCarloSampling(distributions=d, nsamples=5, random_state=843765)
    # Next generate a 3x3 random matrix
    x_mcs2 = MonteCarloSampling(distributions=[d, d, d], nsamples=15, random_state=438975)
    x_mcs_array = x_mcs2.samples.reshape((5, 3, 3))
    # -------------------------------------------------------------------------------------------------------------

    print("Monte Carlo samples of a single random variable from a standard normal distribution.")
    print('Samples stored as an array:')
    print('Data type:', type(x_mcs1.samples))
    print('Number of samples:', len(x_mcs1.samples))
    print('Dimensions of samples:', np.shape(x_mcs1.samples))
    print('Samples')
    print(x_mcs1.samples)
    print()
    print("Monte Carlo samples of a 3x3 matrix of standard normal random variables.")
    print('Samples stored as an array:')
    print('Data type:', type(x_mcs_array))
    print('Number of samples:', len(x_mcs_array))
    print('Dimensions of samples:', np.shape(x_mcs_array))
    print('Samples')
    print(x_mcs_array)
    print()

    # Create a set of samples to be passed into RunModel
    # Here we need to create the mixed samples such that each sample has a single scalar and a single 3x3 matrix.
    # This data structure is essential to passing the input to UQpy correctly.
    x_mixed = []
    for i in range(5):
        x_mixed.append([x_mcs1.samples[i], x_mcs_array[i]])

    print("Combined samples with a scalar and a 3x3 matrix of standard normal random variables.")
    print('Samples stored as a list:')
    print('Data type:', type(x_mixed))
    print('Number of samples:', len(x_mixed))
    print('Dimensions of samples:', np.shape(x_mixed))
    print('Samples')
    print(x_mixed)
    print()

    x_mixed_array = np.atleast_2d(np.asarray(x_mixed))
    print("Combined samples with a scalar and a 3x3 matrix of standard normal random variables.")
    print('Samples stored as ndarray:')
    print('Data type:', type(x_mixed_array))
    print('Number of samples:', len(x_mixed_array))
    print('Dimensions of samples:', np.shape(x_mixed_array))
    print('Samples')
    print(x_mixed_array)
    print()

    # Notice that, in both the ndarray case and the list case, the samples have dimension (5,2). That is, there
    # are five samples of two variables. The first variable is a scalar. The second variable is a 3x3 matrix.

# %% md
#
# 3.1 Pass samples as ndarray, specify format in generated input file, serial execution, do not index samples in
# input_template
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This examples uses the following files:
# - model_script = matlab_model_det.py
# - input_template = prod_determinant.m
# - output_script = process_matlab_output.py

# %%

if pick_model == 'mixed' or pick_model == 'all':
    # Call to RunModel - Here we run the model while instantiating the RunModel object.
    # Notice that we do not specify var_names. This will default to two variables with names x0 and x1. In this
    # case, x0 is a scalar and x1 is a 3x3 matrix. We will read the matrix in without indexing in the
    # input_template.
    t = time.time()
    m = RunModel(samples=x_mixed_array, ntasks=1, model_script='matlab_model_det.py',
                 input_template='prod_determinant.m', model_object_name="matlab",
                 output_script='process_matlab_output.py', output_object_name='read_output',
                 resume=False, model_dir='Matlab_Model', fmt="{:>10.4f}")
    t_ser_matlab = time.time() - t
    print("\nTime for serial execution:")
    print(t_ser_matlab)
    print()
    print("The values returned from the Matlab simulation:")
    print(m.qoi_list)

# %% md
#
# 3.2 Pass samples as ndarray, specify format in generated input file, serial execution, index samples in input_template
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This examples uses the following files:
# - model_script = matlab_model_det_index.py
# - input_template = prod_determinant_index.m
# - output_script = process_matlab_output.py

# %%

if pick_model == 'mixed' or pick_model == 'all':
    # Call to RunModel - Here we run the model while instantiating the RunModel object.
    # Notice that we do not specify var_names. This will default to two variables with names x0 and x1. In this
    # case, x0 is a scalar and x1 is a 3x3 matrix. We will read the matrix in with indexing in the
    # input_template.
    t = time.time()
    m = RunModel(samples=x_mixed_array, ntasks=1, model_script='matlab_model_det_index.py',
                 input_template='prod_determinant_index.m', model_object_name="matlab",
                 output_script='process_matlab_output.py', output_object_name='read_output',
                 resume=False, model_dir='Matlab_Model', fmt="{:>10.4f}")
    t_ser_matlab = time.time() - t
    print("\nTime for serial execution:")
    print(t_ser_matlab)
    print()
    print("The values returned from the Matlab simulation:")
    print(m.qoi_list)

# %% md
#
# 3.3 Pass samples as list, do not specify format in generated input file, parallel execution, do not index samples in
# input_template
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This examples uses the following files:
# - model_script = matlab_model_det.py
# - input_template = prod_determinant.m
# - output_script = process_matlab_output.py

# %%


if pick_model == 'mixed' or pick_model == 'all':
    # Call to RunModel - Here we run the model while instantiating the RunModel object.
    # Notice that we do not specify var_names. This will default to two variables with names x0 and x1. In this
    # case, x0 is a scalar and x1 is a 3x3 matrix. We will read the matrix in without indexing in the
    # input_template.
    t = time.time()
    m = RunModel(samples=x_mixed, ntasks=2, model_script='matlab_model_det.py',
                 input_template='prod_determinant.m', model_object_name="matlab",
                 output_script='process_matlab_output.py', output_object_name='read_output',
                 resume=False, model_dir='Matlab_Model')
    t_ser_matlab = time.time() - t
    print("\nTime for serial execution:")
    print(t_ser_matlab)
    print()
    print("The values returned from the Matlab simulation:")
    print(m.qoi_list)

    # Notice that the solution changes slightly due to the increased precision by not specifying a fmt.

# %% md
#
# 3.4 Pass samples as list, do not specify format in generated input file, parallel execution, index samples in
# input_template
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This examples uses the following files:
# - model_script = matlab_model_det_index.py
# - input_template = prod_determinant_index.m
# - output_script = process_matlab_output.py

# %%

if pick_model == 'mixed' or pick_model == 'all':
    # Call to RunModel - Here we run the model while instantiating the RunModel object.
    # Notice that we do not specify var_names. This will default to two variables with names x0 and x1. In this
    # case, x0 is a scalar and x1 is a 3x3 matrix. We will read the matrix in with indexing in the
    # input_template.
    t = time.time()
    m = RunModel(samples=x_mixed, ntasks=2, model_script='matlab_model_det_index.py',
                 input_template='prod_determinant_index.m', model_object_name="matlab",
                 output_script='process_matlab_output.py', output_object_name='read_output',
                 resume=False, model_dir='Matlab_Model')
    t_ser_matlab = time.time() - t
    print("\nTime for serial execution:")
    print(t_ser_matlab)
    print()
    print("The values returned from the Matlab simulation:")
    print(m.qoi_list)

    # Notice that the solution changes slightly due to the increased precision by not specifying a fmt.

# %% md
#
# 3.5 Pass samples as ndarray, specify format in generated input file, serial execution, partially index samples in
# input_template
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This examples uses the following files:
# - model_script = matlab_model_det_partial.py
# - input_template = prod_determinant_partial.m
# - output_script = process_matlab_output.py

# %%s

if pick_model == 'mixed' or pick_model == 'all':
    # Call to RunModel - Here we run the model while instantiating the RunModel object.
    # Notice that we do not specify var_names. This will default to two variables with names x0 and x1. In this
    # case, x0 is a scalar and x1 is a 3x3 matrix. We will read the matrix in with indexing in the
    # input_template.
    t = time.time()
    m = RunModel(samples=x_mixed_array, ntasks=1, model_script='matlab_model_det_partial.py',
                 input_template='prod_determinant_partial.m', model_object_name="matlab",
                 output_script='process_matlab_output.py', output_object_name='read_output',
                 resume=False, model_dir='Matlab_Model', fmt="{:>10.4f}")
    t_ser_matlab = time.time() - t
    print("\nTime for serial execution:")
    print(t_ser_matlab)
    print()
    print("The values returned from the Python simulation:")
    print(m.qoi_list)