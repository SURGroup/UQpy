"""

Complex probability distribution model
==============================================

Here we define a bivariate probability model, with a dependence structure defined using a gumbel copula. The goal of
inference is to learn the paremeters of the Gaussian marginals and the copula parameter, i.e., the model has 5 unknown
parameters.


"""

#%% md
#
# Initially we have to import the necessary modules.

#%%

import matplotlib.pyplot as plt
from UQpy.inference import DistributionModel, MLE
from UQpy.distributions import Normal
from UQpy.inference import MinimizeOptimizer
from UQpy.distributions import JointIndependent, JointCopula, Gumbel
from UQpy.sampling import ImportanceSampling

#%% md
#
# First data is generated from a true model. A distribution with copulas does not possess a fit method, thus sampling is
# performed using importance sampling/resampling.

#%%

# dist_true exhibits dependence between the two dimensions, defined using a gumbel copula
dist_true = JointCopula(marginals=[Normal(), Normal()], copula=Gumbel(theta=2.))

# generate data using importance sampling: sample from a bivariate gaussian without copula, then weight samples
u = ImportanceSampling(proposal = JointIndependent(marginals=[Normal(), Normal()]),
                       log_pdf_target = dist_true.log_pdf,
                       nsamples=500)
print(u.samples.shape)
print(u.weights.shape)
# Resample to obtain 5,000 data points
u.resample(nsamples=5000)
data_2 = u.unweighted_samples
print('Shape of data: {}'.format(data_2.shape))

fig, ax = plt.subplots()
ax.scatter(data_2[:, 0], data_2[:, 1], alpha=0.2)
ax.set_title('Data points from true bivariate normal with gumbel dependency structure')
plt.show()

#%% md
#
# To define a model for inference, the user must create a custom file, here bivariate_normal_gumbel.py, to compute the
# log_pdf of the distribution, given a bivariate data matrix and a parameter vector of length 5. Note that for any
# probability model that is not one of the simple univariate pdfs supported by UQpy, such a custom file will be
# necessary.

#%%

d_guess = JointCopula(marginals=[Normal(loc=None, scale=None), Normal(loc=None, scale=None)],
                      copula=Gumbel(theta=None))
print(d_guess.get_parameters())
candidate_model = DistributionModel(n_parameters=5, distributions=d_guess)
print(candidate_model.list_params)

#%% md
#
# When calling MLEstimation, the function minimize from the scipy.optimize package is used by default. The user can
# define bounds for the optimization, a seed, the algorithm to be used, and set the algorithm to perform several
# optimization iterations, starting at a different random seed every time.

#%%

optimizer = MinimizeOptimizer(bounds=[[-5, 5], [0, 10], [-5, 5], [0, 10], [1.1, 4]], method="SLSQP")
ml_estimator = MLE(inference_model=candidate_model, data=data_2, optimizer=optimizer)

ml_estimator = MLE(inference_model=candidate_model, data=data_2, optimizer=optimizer,
                   initial_parameters=[1., 1., 1., 1., 4.])

print('ML estimates of the mean={0:.3f} and std. dev={1:.3f} of 1st marginal (true: 0.0, 1.0)'.
      format(ml_estimator.mle[0], ml_estimator.mle[1]))
print('ML estimates of the mean={0:.3f} and std. dev={1:.3f} of 2nd marginal (true: 0.0, 1.0)'.
      format(ml_estimator.mle[2], ml_estimator.mle[3]))
print('ML estimates of the copula parameter={0:.3f} (true: 2.0)'.format(ml_estimator.mle[4]))

#%% md
#
# Again, some known parameters can be fixed during learning.

#%%

d_guess = JointCopula(marginals=[Normal(loc=None, scale=None), Normal(loc=0., scale=1.)],
                      copula=Gumbel(theta=None))
candidate_model = DistributionModel(n_parameters=3, distributions=d_guess)

optimizer = MinimizeOptimizer(bounds=[[-5, 5], [0, 10], [1.1, 4]],
                              method="SLSQP")
ml_estimator = MLE(inference_model=candidate_model, data=data_2, optimizer=optimizer,
                   initial_parameters=[1., 1., 4.])

print('ML estimates of the mean={0:.3f} and std. dev={1:.3f} of 1st marginal (true: 0.0, 1.0)'.
      format(ml_estimator.mle[0], ml_estimator.mle[1]))
print('ML estimates of the copula parameter={0:.3f} (true: 2.0)'.format(ml_estimator.mle[2]))