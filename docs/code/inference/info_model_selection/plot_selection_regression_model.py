"""

Selection between regression models
==============================================

Here candidate models are defined as

.. math:: y=f(θ) + \epsilon

where :math:`f` consists in running RunModel. The three models considered are:

.. math:: f(θ)=θ_{0} x

.. math:: f(θ)=θ_{0} x + θ_{1} x^{2}

.. math:: f(θ)=θ_{0} x + θ_{1} x^{2} + θ_{2} x^{3}


"""

#%% md
#
# Initially we have to import the necessary modules.

#%%
import shutil

from UQpy.inference import DistributionModel, InformationModelSelection, MLE
from UQpy.RunModel import RunModel
import numpy as np
from UQpy.inference import BIC, AIC, AICc
import matplotlib.pyplot as plt
from UQpy.distributions import Gamma, Exponential, ChiSquare, Normal
from UQpy.inference import ComputationalModel


#%% md
#
# First we generate synthetic data using the quadratic model, and add some noise to it.

#%%

param_true = np.array([1.0, 2.0]).reshape((1, -1))
print('Shape of true parameter vector: {}'.format(param_true.shape))

h_func = RunModel(model_script='local_pfn_models.py', model_object_name='model_quadratic', vec=False,
                  var_names=['theta_0', 'theta_1'])
h_func.run(samples=param_true)

# Add noise
error_covariance = 1.
data_clean = np.array(h_func.qoi_list[0])
noise = Normal(loc=0., scale=np.sqrt(error_covariance)).rvs(nsamples=50).reshape((50,))
data_1 = data_clean + noise
print('Shape of data: {}'.format(data_1.shape))

shutil.rmtree(h_func.model_dir)

#%% md
#
# Create instances of the Model class for three models: linear, quadratic and cubic

#%%

names = ['linear', 'quadratic', 'cubic']
estimators = []
for i in range(3):
    h_func = RunModel(model_script='local_pfn_models.py', model_object_name='model_' + names[i], vec=False,
                      var_names=['theta_{}'.format(j) for j in range(i + 1)])
    M = ComputationalModel(runmodel_object=h_func, n_parameters=i + 1,
                           name=names[i], error_covariance=error_covariance)
    estimators.append(MLE(inference_model=M, data=data_1))

#%% md
#
# Apart from the data, candidate models and method (BIC, AIC...), InfoModelSelection also takes as inputs lists of
# inputs to the maximum likelihood class (iter_optim, method_optim, ...). Those inputs should be lists of length
# len(candidate_models).

#%%

from UQpy.utilities.MinimizeOptimizer import MinimizeOptimizer
optimizer = MinimizeOptimizer(method='nelder-mead')
selector = InformationModelSelection(parameter_estimators=estimators, criterion=BIC(), n_optimizations=[1]*3)
selector.sort_models()
print('Sorted models: ', [m.name for m in selector.candidate_models])
print('Values of criterion: ', selector.criterion_values)
print('Values of data fit:', [cr - pe for (cr, pe) in zip(selector.criterion_values, selector.penalty_terms)])
print('Values of penalty term (complexity):', selector.penalty_terms)
print('Values of model probabilities:', selector.probabilities)

#%% md
#
# Plot the results

#%%

domain = np.linspace(0, 10, 50)
fig, ax = plt.subplots(figsize=(8, 6))

for i, (model, estim) in enumerate(zip(selector.candidate_models, selector.parameter_estimators)):
    model.runmodel_object.run(samples=estim.mle.reshape((1, -1)), append_samples=False)
    y = model.runmodel_object.qoi_list[-1].reshape((-1,))
    ax.plot(domain, y, label=selector.candidate_models[i].name)
    shutil.rmtree(model.runmodel_object.model_dir)

plt.plot(domain, data_1, linestyle='none', marker='.', label='data')
plt.xlabel('x')
plt.ylabel('y')

plt.legend()
plt.show()



#%% md
#
# For this case, one can observe that both the quadratic and cubic model are capable of explaining the data. The cubic
# model is penalized due to its higher complexity (penalty_term) and thus the quadratic model is preferred.
