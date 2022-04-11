"""

Rosenbrock distribution
============================================

"""

# %% md
#
# For importance sampling, the function must be written in a way that it can
# evaluate multiple samples at once.

# %%

from UQpy.distributions import Uniform, JointIndependent
from UQpy.sampling import ImportanceSampling
import time
import matplotlib.pyplot as plt
import numpy as np


def log_Rosenbrock(x, param):
    return (-(100 * (x[:, 1] - x[:, 0] ** 2) ** 2 + (1 - x[:, 0]) ** 2) / param)


proposal = JointIndependent([Uniform(loc=-8, scale=16), Uniform(loc=-10, scale=60)])
print(proposal.get_parameters())

# %% md
#
# Run IS
# -------

# %%

t4 = time.time()

w = ImportanceSampling(log_pdf_target=log_Rosenbrock, args_target=(20,), proposal=proposal, nsamples=10000)

t_IS = time.time() - t4
print(t_IS)

w.resample(nsamples=1000)
plt.plot(w.unweighted_samples[:, 0], w.unweighted_samples[:, 1], 'gs', alpha=0.2)
print(w.unweighted_samples.shape)
plt.legend(['IS'])
plt.show()

# %% md
#
# Run IS by adding samples: call the run method in a loop (one can also look at diagnostics)
# ----------------------------------------------------------------------------------------------

# %%

t4 = time.time()

w = ImportanceSampling(log_pdf_target=log_Rosenbrock, args_target=(20,), proposal=proposal)
for nsamples in [5000, 5000, 5000, 5000]:
    w.run(nsamples)
    print(w.samples.shape)
    # IS_diagnostics(weights=w.weights, graphics=False)
t_IS = time.time() - t4
print(t_IS)

w.resample(nsamples=1000)
plt.plot(w.unweighted_samples[:, 0], w.unweighted_samples[:, 1], 'gs', alpha=0.2)
plt.legend(['IS'])
plt.show()

# %% md
#
# Another example: sampling from a bivariate with copula dependence. Giving a random state enforces that results are
# the same for repeatability.

# %%

from UQpy.distributions import Normal, Gumbel, JointCopula

dist_true = JointCopula(marginals=[Normal(), Normal()], copula=Gumbel(theta=2.))
proposal1 = JointIndependent(marginals=[Normal(), Normal()])

sampler = ImportanceSampling(log_pdf_target=dist_true.log_pdf, proposal=proposal1, random_state=123,
                             nsamples=500)
print(sampler.samples.shape)
print(sampler.weights.shape)
print(np.round(sampler.samples[-5:], 4))
