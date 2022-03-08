"""

Selection between univariate distributions
==============================================

Here a model is defined that is of the form

.. math:: y=f(\theta) + \epsilon

where f consists in running RunModel. In particular, here $f(\theta)=\theta_{0} x + \theta_{1} x^{2}$ is a regression model.


"""

#%% md
#
# Initially we have to import the necessary modules.

#%%

from UQpy.inference import DistributionModel, InformationModelSelection, MLE
import numpy as np
import matplotlib.pyplot as plt
from UQpy.distributions import Gamma, Exponential, ChiSquare

#%% md
#
# Generate data using a gamma distribution.

#%%

data = Gamma(a=2, loc=0, scale=2).rvs(nsamples=500, random_state=12)
print(data.shape)

#%% md
#
# Define the models to be compared, then call InfoModelSelection to perform model selection. By default,
# InfoModelSelection returns its outputs, fitted parameters, value of the chosen criteria, model probabilities and so
# on, in a sorted order, i.e., starting with the most probable model. However, if setting sorted_outputs=False, the
# class output attributes are given in the same order as the candidate_models.

#%%


# Define the models to be compared, for each model one must create an instance of the model class

m0 = DistributionModel(distributions=Gamma(a=None, loc=None, scale=None), n_parameters=3, name='gamma')
m1 = DistributionModel(distributions=Exponential(loc=None, scale=None), n_parameters=2, name='exponential')
m2 = DistributionModel(distributions=ChiSquare(df=None, loc=None, scale=None), n_parameters=3, name='chi-square')

candidate_models = [m0, m1, m2]

mle1 = MLE(inference_model=m0, random_state=0, data=data)
mle2 = MLE(inference_model=m1, random_state=0, data=data)
mle3 = MLE(inference_model=m2, random_state=0, data=data)

#%% md
#
# Perform model selection using different information criteria

#%%

from UQpy.inference import BIC, AIC, AICc

criteria = [BIC(), AIC(), AICc()]
sorted_names =[]
criterion_value = []
param =[]
for criterion in criteria:
    selector = InformationModelSelection(parameter_estimators=[mle1, mle2, mle3], criterion=criterion,
                                         n_optimizations=[5]*3)
    selector.sort_models()
    print('Sorted model using ' + str(criterion) + ' criterion: ' + ', '.join(
        m.name for m in selector.candidate_models))
    if isinstance(criterion, BIC):
        criterion_value = selector.criterion_values
        sorted_names = [m.name for m in selector.candidate_models]
        param = [m.mle for m in selector.parameter_estimators]

width = 0.5
ind = np.arange(len(sorted_names))
p1 = plt.bar(ind, criterion_value, width=width)
# p2 = plt.bar(ind, criterion_value-data_fit_value, bottom=data_fit_value, width = width)

plt.ylabel('BIC criterion')
plt.title('Model selection using BIC criterion: model fit vs. Ockam razor')
plt.xticks(ind, sorted_names)

plt.show()

print('Shape parameter of the gamma distribution: {}'.format(param[sorted_names.index('gamma')][0]))
print('DoF of the chisquare distribution: {}'.format(param[sorted_names.index('chi-square')][0]))


#%% md
#
# Note that here both the chisquare and gamma are capable of explaining the data, with :math:`a = \nu/2`, :math:`a` is
# gamma's shape parameter and :math:`\nu` is the number of DOFs in chi-square distribution.
