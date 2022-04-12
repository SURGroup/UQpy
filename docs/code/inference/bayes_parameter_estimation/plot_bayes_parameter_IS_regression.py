"""

Parameter estimation using Importance Sampling - Regression Model
=============================================================================

Here a model is defined that is of the form

.. math:: y=f(θ) + \epsilon

where :math:`f` consists in running RunModel. In particular, here :math:`f(θ)= θ_{0} x + θ_{1} x^{2}` is
a regression model.

"""

#%% md
#
# Initially we have to import the necessary modules.

#%%
import shutil

import numpy as np
import matplotlib.pyplot as plt

from UQpy import PythonModel
from UQpy.sampling.ImportanceSampling import ImportanceSampling
from UQpy.inference import BayesParameterEstimation, ComputationalModel
from UQpy.run_model.RunModel import RunModel  # required to run the quadratic model
from sklearn.neighbors import KernelDensity  # for the plots
from UQpy.distributions import JointIndependent, Normal

def pdf_from_kde(domain, samples1d):
    bandwidth = 1.06 * np.std(samples1d) * samples1d.size ** (-1 / 5)
    kde = KernelDensity(bandwidth=bandwidth).fit(samples1d.reshape((-1, 1)))
    log_dens = kde.score_samples(domain)
    return np.exp(log_dens)

#%% md
#
# First we generate synthetic data, and add some noise to it.

#%%

param_true = np.array([1.0, 2.0]).reshape((1, -1))
print('Shape of true parameter vector: {}'.format(param_true.shape))

model = PythonModel(model_script='local_pfn_models.py', model_object_name='model_quadratic', delete_files=True,
                    var_names=['theta_0', 'theta_1'])
h_func = RunModel(model=model)
h_func.run(samples=param_true)

# Add noise
error_covariance = 1.
data_clean = np.array(h_func.qoi_list[0])
noise = Normal(loc=0., scale=np.sqrt(error_covariance)).rvs(nsamples=50).reshape((50,))
data_3 = data_clean + noise
print('Shape of data: {}'.format(data_3.shape))


inference_model = ComputationalModel(n_parameters=2, runmodel_object=h_func, error_covariance=error_covariance)

sampling = ImportanceSampling(proposal=JointIndependent([Normal(scale=2, )] * 2))
bayes_estimator =\
    BayesParameterEstimation(inference_model=inference_model,
                             data=data_3,
                             sampling_class=sampling,
                             nsamples=5000)

s = bayes_estimator.sampler.samples
w = bayes_estimator.sampler.weights
print(sum(w))

# print results
fig, ax = plt.subplots(1, 2)
for i in range(2):
    ax[i].hist(x=s[:, i], weights=None, density=True, range=(-4, 4), bins=20, color='blue', alpha=0.4,
               label='prior')
    ax[i].hist(x=s[:, i], weights=w, density=True, range=(-4, 4), bins=20, color='orange', alpha=0.7,
               label='posterior')
    ax[i].legend()
    ax[i].set_title('theta {}'.format(i + 1))

plt.show()

