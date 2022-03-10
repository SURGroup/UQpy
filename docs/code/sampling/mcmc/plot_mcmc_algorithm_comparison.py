"""

Comparison of various MCMC aglorithms
============================================================

This script illustrates performance of various MCMC algorithms currently integrated in UQpy:
- Metropolis Hastings (MH)
- Modified Metropolis Hastings (MMH)
- Affine Invariant with Stretch moves (Stretch)
- Adaptive Metropolis with delayed rejection (DRAM)
"""

# %% md
#
# Import the necessary libraries.

# %%

import numpy as np
import matplotlib.pyplot as plt
import time

from UQpy.sampling import MetropolisHastings, Stretch, ModifiedMetropolisHastings, DREAM, DRAM

# %% md
# Affine invariant with Stretch moves
# -----------------------------------
# This algorithm requires as seed a few samples near the region of interest. Here MH is first run to obtain few samples,
# used as seed within the Stretch algorithm.

# %%

def log_Rosenbrock(x):
    return (-(100 * (x[:, 1] - x[:, 0] ** 2) ** 2 + (1 - x[:, 0]) ** 2) / 20)

x = MetropolisHastings(dimension=2, burn_length=0, jump=10, n_chains=1, log_pdf_target=log_Rosenbrock,
                       nsamples=5000)
print(x.samples.shape)
plt.figure()
plt.plot(x.samples[:, 0], x.samples[:, 1], 'o')
plt.show()

x = Stretch(burn_length=0, jump=10, log_pdf_target=log_Rosenbrock, seed=x.samples[:10].tolist(), scale=2.,
            nsamples=5000)
print(x.samples.shape)

plt.figure()
plt.plot(x.samples[:, 0], x.samples[:, 1], 'o')
plt.show()

# %% md
# DREAM algorithm: compare with MH (inputs parameters are set as their default values)
# -----------------------------------

# %%

# Define a function to sample seed uniformly distributed in the 2d space ([-20, 20], [-4, 4])
from UQpy.distributions import Uniform, JointIndependent
prior_sample = lambda nsamples: np.array([[-2, -2]]) + np.array([[4, 4]]) * JointIndependent(
    [Uniform(), Uniform()]).rvs(nsamples=nsamples)

fig, ax = plt.subplots(ncols=2, figsize=(12, 4))
seed = prior_sample(nsamples=7)

x = MetropolisHastings(dimension=2, burn_length=500, jump=50, seed=seed.tolist(),
                       log_pdf_target=log_Rosenbrock, nsamples=1000)
ax[0].plot(x.samples[:, 0], x.samples[:, 1], 'o')

x = DREAM(dimension = 2, burn_length = 500, jump = 50, seed = seed.tolist(), log_pdf_target = log_Rosenbrock,
          nsamples=1000)
ax[1].plot(x.samples[:, 0], x.samples[:, 1], 'o')

plt.show()


# %% md
# DRAM algorithm
# -----------------------------------

# %%

fig, ax = plt.subplots(ncols=2, figsize=(12, 4))
t = time.time()
seed = prior_sample(nsamples=1)

x = MetropolisHastings(dimension = 2, burn_length = 500, jump = 10, seed = seed.tolist(), log_pdf_target = log_Rosenbrock,
                       nsamples=1000)

ax[0].plot(x.samples[:, 0], x.samples[:, 1], 'o')
ax[0].set_xlabel('x1')
ax[0].set_ylabel('x2')
ax[0].set_title('algorithm: MH')
print('time to run MH' + ': {}s'.format(time.time() - t))

x = DRAM(dimension=2, burn_length=500, jump=10, seed=seed.tolist(), log_pdf_target=log_Rosenbrock,
         save_covariance=True, nsamples=1000)

ax[1].plot(x.samples[:, 0], x.samples[:, 1], 'o')
ax[1].set_xlabel('x1')
ax[1].set_ylabel('x2')
ax[1].set_title('algorithm: DRAM' )
print('time to run DRAM'  + ': {}s'.format(time.time() - t))

plt.show()

# look at the covariance adaptivity
fig, ax = plt.subplots()
adaptive_covariance = np.array(x.adaptive_covariance)
for i in range(2):
    ax.plot(np.sqrt(adaptive_covariance[:, 0, i, i]), label='dimension {}'.format(i))
ax.set_title('Adaptive proposal std. dev. in both dimensions')
ax.legend()
plt.show()

# %% md
# MMH: target pdf is given as a joint pdf
# -----------------------------------
# The target pdf should be a 1 dimensional distribution or set of 1d distributions.

# %%

from UQpy.distributions import Normal
proposal = [Normal(), Normal()]
proposal_is_symmetric = [False, False]

x = ModifiedMetropolisHastings(dimension=2, burn_length=500, jump=50, log_pdf_target=log_Rosenbrock,
                               proposal=proposal, proposal_is_symmetric=proposal_is_symmetric, n_chains=1,
                               nsamples=500)

fig, ax = plt.subplots()
ax.plot(x.samples[:, 0], x.samples[:, 1], linestyle='none', marker='.')

# %% md
# MMH: target pdf is given as a couple of independent marginals
# -----------------------------------

# %%

log_pdf_target = [Normal(loc=0., scale=5.).log_pdf, Normal(loc=0., scale=20.).log_pdf]

proposal = [Normal(), Normal()]
proposal_is_symmetric = [True, True]

x = ModifiedMetropolisHastings(dimension = 2, burn_length = 100, jump = 10, log_pdf_target = log_pdf_target,
                               proposal = proposal, proposal_is_symmetric = proposal_is_symmetric, n_chains = 1,
                               nsamples=1000)

fig, ax = plt.subplots()
ax.plot(x.samples[:, 0], x.samples[:, 1], linestyle='none', marker='.')
plt.show()
print(x.samples.shape)

# %% md
# Use random_state to provide repeated results
# -----------------------------------

# %%

from UQpy.distributions import Normal, Gumbel, JointCopula, JointIndependent, Uniform
seed = Uniform().rvs(nsamples=2 * 10).reshape((10, 2))
dist_true = JointCopula(marginals=[Normal(), Normal()], copula=Gumbel(theta=2.))
proposal = JointIndependent(marginals=[Normal(scale=0.2), Normal(scale=0.2)])

for _ in range(3):
    sampler = ModifiedMetropolisHastings(log_pdf_target=dist_true.log_pdf, proposal=proposal, seed=[0., 0.],
                                         random_state=123, nsamples=500)
    print(sampler.samples.shape)
    print(np.round(sampler.samples[-5:], 4))

plt.plot(sampler.samples[:, 0], sampler.samples[:, 1], linestyle='none', marker='+')
