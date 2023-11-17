"""

Python
==================================
"""






# %% md
#
# The RunModel class is capable of passing input in different formats into a single computational model. This means that
# the samples passed into a model can be passed as:
#
# - floating point values
# - numpy arrays
# - lists
# - tuples
# - lists of other iterables
# - numpy arrays of other iterables
# - or any combination of the above
#
# In the examples below, we demonstrate the use of a Python computational model with inputs that are combinations of the
# above.
#
# Some notes on their use:
#
# 1. UQpy converts all sample input to a numpy array with at least two dimensions. The first dimension,
# i.e. len(samples) must correspond to the number of samples being passed for model execution. The second dimension,
# i.e. len(samples[0]) must correspond to the number of variables that each sample possesses.
#
# 2. Each individual sample, i.e. sample[j], may be composed of multiple data types -- with each variable having a
# different data type. For example, sample[j][k] may be a floating point value and sample[j][l] may be an array of
# arbitrary dimension.
#
# 3. If a specific variable has multiple dimensions, the user may specify the index to be return in the input file.
# For example, the placeholder for a variable x1 corresponding to sample[j][l] that is an array of shape (1,4) can be
# read as <x1[0, 3]>, which will return the final (0,3) component of samples[j][l].
#
# 4. If the user does not specify the index for a multidimensional variable, then the entire multidimensional variable
# is flattened and written with comma delimiters.

# %%

# %% md
#
# Python Model Summary
# --------------------
# Examples 1-2:
# The provided Matlab models take the sum of three random variables:
#
# .. math:: s = \sum_{i=1}^3 x_i
#
# .. math:: x_i \sim N(0,1)
#
# Example 3:
# The provided Matlab model takes the product of a random variable and the determinant of a random matrix:
#
# .. math:: z = x \det(Y)
#
# .. math:: x \sim N(0,1)
#
# :math:`y` is a 3x3 matrix of standard normal random variables.
#
# The Python model may be provided as either a class or a function. The examples below explore both cases.

# %%

from UQpy.sampling import MonteCarloSampling
from UQpy.run_model.RunModel import RunModel
from UQpy.run_model.model_types.PythonModel import PythonModel
from UQpy.distributions import Normal
import time
import numpy as np


# %%

# %% md
#
# Pick which model to run
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Options:
# - 'all'
# - 'scalar'
# - 'vector'
# - 'mixed'
# - 'fixed'

# %%

pick_model = 'all'


# %% md
#
# Example 1: Three scalar random variables
# ----------------------------------------------------
# In this example, we pass three scalar random variables. Note that this is different from assigning a single variable
# with three components, which will be handled in the following example.
#
# Here we will pass the samples both as an ndarray and as a list. Recall that UQpy converts all samples into an ndarray
# of at least two dimensions internally.

# %%

if pick_model in {'scalar', 'vector', 'all'}:
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
# 1.1 Pass samples as ndarray, Python class called, serial execution
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This examples uses the following files:
# - model_script = python_model.py

# %%

if pick_model in {'scalar', 'all'}:
    # Call to RunModel - Here we run the model while instantiating the RunModel object.
    t = time.time()
    m = PythonModel(model_script='python_model.py', model_object_name='SumRVs')
    m11 = RunModel(model=m, ntasks=1 )
    m11.run(samples=x_mcs.samples,)
    t_ser_python = time.time() - t
    print("\nTime for serial execution:")
    print(t_ser_python)
    print()
    print("The values returned from the Matlab simulation:")
    print(m11.qoi_list)

# %% md
#
# 1.2 Pass samples as list, Python function called, parallel execution
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This examples uses the following files:
# - model_script = python_model.py

# %%

if pick_model in {'scalar', 'all'}:
    # Call to RunModel - Here we run the model while instantiating the RunModel object.
    t = time.time()
    m = PythonModel(model_script='python_model.py', model_object_name='sum_rvs')
    m12 = RunModel(model=m, samples=x_mcs_list, ntasks=2)
    t_par_python = time.time() - t
    print("\nTime for parallel execution:")
    print(t_par_python)
    print()
    print("The values returned from the Matlab simulation:")
    print(m12.qoi_list)


# %% md
#
# Example 2: Single tri-variate random variable
# -----------------------------------------------------
# In this example, we pass three random variables in as a trivariate random variable. Note that this is different from
# assigning three scalar random variables, which was be handled in Example 1.
#
# Again, we will pass the samples both as an ndarray and as a list. Recall that UQpy converts all samples into an
# ndarray of at least two dimensions internally.

# %%

# %% md
#
# Restructure the samples
# -----------------------------------------------------
# To pass the samples in as a single tri-variate variable, we need reshape the samples from shape (5, 3) to
# shape (5, 1, 3)

# %%

if pick_model in {'vector', 'all'}:
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
# 2.1 Pass samples as ndarray, Python function called, serial execution
# -----------------------------------------------------------------------
# This example uses the following files:
# - model_script = python_model.py

# %%

if pick_model in {'vector', 'all'}:
    # Call to RunModel - Here we run the model while instantiating the RunModel object.
    t = time.time()
    m=PythonModel(model_script='python_model.py', model_object_name='sum_rvs_vec')
    m21 = RunModel(samples=x_mcs_tri, ntasks=1,  model=m)
    t_ser_python = time.time() - t
    print("\nTime for serial execution:")
    print(t_ser_python)
    print()
    print("The values returned from the Matlab simulation:")
    print(m21.qoi_list)

# %% md
#
# 2.2 Pass samples as list, Python class called, parallel execution
# --------------------------------------------------------------------
# This example uses the following files:
# - model_script = python_model.py

# %%

if pick_model == 'vector' or pick_model == 'all':
    # Call to RunModel - Here we run the model while instantiating the RunModel object.
    t = time.time()
    m=PythonModel(model_script='python_model.py', model_object_name='SumRVs')
    m22 = RunModel(samples=x_mcs_tri_list, ntasks=2, model=m)
    t_par_python = time.time() - t
    print("\nTime for parallel execution:")
    print(t_par_python)
    print()
    print("The values returned from the Matlab simulation:")
    print(m22.qoi_list)

# %% md
#
# Example 3: Passing a scalar and an array to RunModel
# -----------------------------------------------------
# In this example, we pass a single scalar random variable as well as an array into a Matlab model.
#
# Again, we will pass the samples both as an ndarray and as a list. Recall that UQpy converts all samples into an
# ndarray of at least two dimensions internally.

# %%

if pick_model == 'mixed' or pick_model == 'vector' or pick_model == 'all':
    # Call MCS to generate samples
    # THIS WILL NEED TO BE REWRITTEN WITH DISTRIBUTION AND MCS UPDATES --------------------------------------------
    # First generate the scalar random variable
    #     x_mcs1 = MCS(dist_name=['Normal'], dist_params=[[0,1]], nsamples=5, var_names = ['var1'])
    # Next generate a 3x3 random matrix
    #     x_mcs2 = MCS(dist_name=['Normal','Normal','Normal'], dist_params=[[0,1],[0,1],[0,1]], nsamples=15)
    #     x_mcs_array = x_mcs2.samples.reshape((5,3,3))
    # -------------------------------------------------------------------------------------------------------------

    # Call MCS to generate samples
    d = Normal(loc=0, scale=1)
    x_mcs1 = MonteCarloSampling(distributions=d, nsamples=5, random_state=987979)
    x_mcs2 = MonteCarloSampling(distributions=[d, d, d], nsamples=15, random_state=34876)
    x_mcs_array = x_mcs2.samples.reshape((5, 3, 3))

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
# 3.1 Pass samples as ndarray, Python class called, serial execution
# ---------------------------------------------------------------------
# This examples uses the following files:
# - model_script = python_model.py

# %%

if pick_model == 'mixed' or pick_model == 'all':
    # Call to RunModel - Here we run the model while instantiating the RunModel object.
    t = time.time()
    m=PythonModel(model_script='python_model.py', model_object_name='DetRVs')
    m31 = RunModel(samples=x_mixed_array, ntasks=1, model=m)
    t_ser_python = time.time() - t
    print("\nTime for serial execution:")
    print(t_ser_python)
    print()
    print("The values returned from the Matlab simulation:")
    print(m31.qoi_list)

# %% md
#
# 3.2 Pass samples as list, Python function called, parallel execution
# ------------------------------------------------------------------------
# This examples uses the following files:
# - model_script = python_model.py

# %%

if pick_model == 'mixed' or pick_model == 'all':
    # Call to RunModel - Here we run the model while instantiating the RunModel object.
    # Note that the parallel model_object handles only one sample at a time.
    t = time.time()
    m=PythonModel(model_script='python_model.py', model_object_name='det_rvs_par')
    m32 = RunModel(samples=x_mixed, ntasks=1, model=m)
    t_par_python = time.time() - t
    print("\nTime for parallel execution:")
    print(t_par_python)
    print()
    print("The values returned from the Matlab simulation:")
    print(m32.qoi_list)

# %% md
#
# Example 4: Passing a fixed variable and an array of Random Variables to RunModel
# -----------------------------------------------------------------------------------
# In this example, we pass a fixed value coefficient as well as an array of random variables into a Python model.
#
# Again, we will pass the samples both as an ndarray and as a list. Recall that UQpy converts all samples into an
# ndarray of at least two dimensions internally.

# %%

if pick_model == 'mixed' or pick_model == 'all':
    x = 2.5

    print('Constant Coefficient:')
    print(x)
    print()

    print("Monte Carlo samples of a 3x3 matrix of standard normal random variables.")
    print('Samples stored as an array:')
    print('Data type:', type(x_mcs_array))
    print('Number of samples:', len(x_mcs_array))
    print('Dimensions of samples:', np.shape(x_mcs_array))
    print('Samples')
    print(x_mcs_array)
    print()

    x_mcs_list = list(x_mcs_array)

    print("3x3 matrix of standard normal random variables.")
    print('Samples stored as ndarray:')
    print('Data type:', type(x_mcs_list))
    print('Number of samples:', len(x_mcs_list))
    print('Dimensions of samples:', np.shape(x_mcs_list))
    print('Samples')
    print(x_mcs_list)
    print()

    # Notice that, in both the ndarray case and the list case, the samples have dimension (5,2). That is, there
    # are five samples of two variables. The first variable is a scalar. The second variable is a 3x3 matrix.

# %% md
#
# 4.1 Pass samples as ndarray, Python class called, serial execution
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This examples uses the following files:
# - model_script = python_model.py

if pick_model == 'mixed' or pick_model == 'all':
    # Call to RunModel - Here we run the model while instantiating the RunModel object.
    t = time.time()
    m=PythonModel(model_script='python_model.py', model_object_name='det_rvs_fixed')
    m41 = RunModel(samples=x_mcs_array, ntasks=1, model=m, coeff=x)
    t_ser_python = time.time() - t
    print("\nTime for serial execution:")
    print(t_ser_python)
    print()
    print("The values returned from the Matlab simulation:")
    print(m41.qoi_list)

# %% md
#
# 4.2 Pass samples as list, Python class called, serial execution
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This examples uses the following files:
# - model_script = python_model.py

if pick_model == 'mixed' or pick_model == 'all':
    # Call to RunModel - Here we run the model while instantiating the RunModel object.
    t = time.time()
    m=PythonModel(model_script='python_model.py', model_object_name='det_rvs_fixed')
    m42 = RunModel(samples=x_mcs_list, ntasks=1, model=m, coeff=x)
    t_ser_python = time.time() - t
    print("\nTime for serial execution:")
    print(t_ser_python)
    print()
    print("The values returned from the Matlab simulation:")
    print(m42.qoi_list)