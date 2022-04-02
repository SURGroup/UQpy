"""

SROM on a Gamma distribution
======================================================================

In this example, Stratified sampling is used to generate samples from Gamma distribution and weights are defined using
Stochastic Reduce Order Model (SROM).
"""

#%% md
#
# Import the necessary libraries. Here we import standard libraries such as numpy and matplotlib, but also need to
# import the :class:`.TrueStratifiedSampling` and :class:`.SROM` class from UQpy.

#%%

from UQpy.surrogates import SROM
from UQpy.sampling import RectangularStrata
from UQpy.sampling import TrueStratifiedSampling
from UQpy.distributions import Gamma
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

#%% md
#
# Create a distribution object for :class:`.Gamma` distribution with shape, shift and scale parameters as :math:`2`,
# :math:`1` and :math:`3`.

#%%

marginals = [Gamma(a=2., loc=1., scale=3.), Gamma(a=2., loc=1., scale=3.)]

#%% md
#
# Create a strata object.

#%%

strata = RectangularStrata(strata_number=[4, 4])

#%% md
#
# Using UQpy :class:`.TrueStratifiedSampling` class to generate samples for two random variables having :class:`.Gamma`
# distribution.

#%%

x = TrueStratifiedSampling(distributions=marginals, strata_object=strata, nsamples_per_stratum=1)

#%% md
#
# Run :class:`.SROM` using the defined :class:`.Gamma` distribution. Here we use the following parameters.
#
# - :class:`.Gamma` distribution with shape, shift and scale parameters as :math:`2`, :math:`1` and :math:`3`.
# - First and second order moments about origin are :math:`6` and :math:`54`.
# - Notice that :code:`pdf_target` references the :class:`.Gamma` function directly and does not designate it as a string.
# - Samples are uncorrelated, i.e. also default value of correlation.

#%%

y = SROM(samples=x.samples,
         target_distributions=marginals,
         moments=np.array([[6., 6.], [54., 54.]]))
y.run(properties=[True, True, True, True])

#%% md
#
# Plot the samples and weights from :class:`.SROM` class. Also, compared with the CDF of gamma distribution.

#%%

c = np.concatenate((y.samples, y.sample_weights.reshape(y.sample_weights.shape[0], 1)), axis=1)
d = c[c[:, 0].argsort()]
plt.plot(d[:, 0], np.cumsum(d[:, 2], axis=0), 'o')
plt.plot(np.arange(1, 15, 0.1), stats.gamma.cdf(np.arange(1, 15, 0.1), 2, loc=1, scale=3))
plt.legend(['RV1_SROM', 'CDF'])
plt.show()
e = c[c[:, 1].argsort()]
plt.plot(e[:, 1], np.cumsum(e[:, 2], axis=0), 'o')
plt.plot(np.arange(1, 15, 0.1), stats.gamma.cdf(np.arange(1, 15, 0.1), 2, loc=1, scale=3))
plt.legend(['RV2_SROM', 'CDF'])
plt.show()


#%% md
#
# A note on the weights corresponding to error in distribution, moments and correlation of random variables:
#
# - For this illustration, error_weights are not defined and default value is :math:`[1, 0.2, 0]`. These weights can be changed to obtain desired accuracy in certain properties.

#%%

print(y.sample_weights)
