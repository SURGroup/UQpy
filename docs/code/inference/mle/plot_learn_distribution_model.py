"""

Simple probability distribution model
==============================================

In the following we learn the mean and covariance of a univariate gaussian distribution from data.
"""

#%% md
#
# Initially we have to import the necessary modules.

#%%

import numpy as np
import matplotlib.pyplot as plt
from UQpy.inference import DistributionModel, MLE
from UQpy.distributions import Normal
from UQpy.inference import MinimizeOptimizer

#%% md
#
# First, for the sake of this example, we generate fake data from a gaussian distribution with mean 0 and
# standard deviation 1.

#%%

mu, sigma = 0, 0.1  # true mean and standard deviation
data_1 = np.random.normal(mu, sigma, 1000).reshape((-1, 1))
print('Shape of data vector: {}'.format(data_1.shape))

count, bins, ignored = plt.hist(data_1, 30, density=True)
plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)),
         linewidth=2, color='r')
plt.title('Histogram of the data')
plt.show()

#%% md
#
# Create an instance of the class Model. The user must define the number of parameters to be estimated, in this case 2
# (mean and standard deviation), and set those parameters to be learnt as None when instantiating the Distribution
# object. For maximum likelihood estimation, no prior pdf is required.

#%%

# set parameters to be learnt as None
dist = Normal(loc=None, scale=None)
candidate_model = DistributionModel(n_parameters=2, distributions=dist)

ml_estimator = MLE(inference_model=candidate_model, data=data_1, n_optimizations=3)
print('ML estimates of the mean={0:.3f} (true=0.) and std. dev={1:.3f} (true=0.1)'.format(
    ml_estimator.mle[0], ml_estimator.mle[1]))

#%% md
#
# We can also fix one of the parameters and learn the remaining one

#%%

d = Normal(loc=0., scale=None)
candidate_model = DistributionModel(n_parameters=1, distributions=d)

optimizer = MinimizeOptimizer(bounds=[[0.0001, 2.]])
ml_estimator = MLE(inference_model=candidate_model, data=data_1,
                   n_optimizations=1)
print('ML estimates of the std. dev={0:.3f} (true=0.1)'.format(ml_estimator.mle[0]))