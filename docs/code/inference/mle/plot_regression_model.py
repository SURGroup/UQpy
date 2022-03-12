"""

Regression model
==============================================

Here a model is defined that is of the form

.. math:: y=f(\theta) + \epsilon

where f consists in running RunModel. In particular, here $f(\theta)=\theta_{0} x + \theta_{1} x^{2}$ is a regression model.


"""

#%% md
#
# Initially we have to import the necessary modules.

#%%
import shutil

import numpy as np
from UQpy.inference import ComputationalModel, MLE
from UQpy.distributions import Normal
from UQpy.inference import MinimizeOptimizer
from UQpy.RunModel import RunModel

#%% md
#
# First we generate synthetic data, and add some noise to it.

#%%

# Generate data

param_true = np.array([1.0, 2.0]).reshape((1, -1))
print('Shape of true parameter vector: {}'.format(param_true.shape))

h_func = RunModel(model_script='local_pfn_models.py', model_object_name='model_quadratic', vec=False,
                  var_names=['theta_0', 'theta_1'])
h_func.run(samples=param_true)

# Add noise
error_covariance = 1.
data_clean = np.array(h_func.qoi_list[0])
noise = Normal(loc=0., scale=np.sqrt(error_covariance)).rvs(nsamples=50).reshape((50,))
data_3 = data_clean + noise
print('Shape of data: {}'.format(data_3.shape))


#%% md
#
# Then we create an instance of the Model class, using model_type='python', and we perform maximum likelihood estimation
# of the two parameters.

#%%

candidate_model = ComputationalModel(n_parameters=2, runmodel_object=h_func, error_covariance=error_covariance)

optimizer = MinimizeOptimizer(method='nelder-mead')
ml_estimator = MLE(inference_model=candidate_model, data=data_3, n_optimizations=1)
print('fitted parameters: theta_0={0:.3f} (true=1.), and theta_1={1:.3f} (true=2.)'.format(
    ml_estimator.mle[0], ml_estimator.mle[1]))

shutil.rmtree(h_func.model_dir)