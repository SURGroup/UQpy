"""

Correlated Normal Random Variables with a Linear Performance Function
=========================================================================

This example runs subset simulation for a linear performance function having a specified reliability index
:math:`β` and standard normal random variables with specified correlation using the affine invariate ensemble
"stretch" sampler and the conventional Modified Metropolis Hastings algorithm.

For more details, refer to:

- Shields, M.D. and Giovanis, D.G. and Sundar, V.S. "Subset Simulation for problems with strongly non-Gaussian, highly anisotropic, and degenerate distributions," Computers and Structures. (In Review)
"""

# %% md
#
# 1. Import the necessary libraries

# %%
import shutil

from UQpy.reliability import SubsetSimulation
import matplotlib.pyplot as plt
from UQpy.sampling import ModifiedMetropolisHastings, Stretch
import time
import numpy as np
from UQpy.distributions import MultivariateNormal
from UQpy.RunModel import RunModel
import scipy.stats as stats

# %% md
#
# 2. Define the reliability index, correlation, dimension, algorithm, and the number of trials to run.

# %%

# Specified Reliability Index
beta = 4

# Specified Correlation
rho = 0.5

# Dimension
dim = 2

# Specify the number of trials to run
ntrials = 1

# Define the correlation matrix
C = np.ones((dim, dim)) * rho
np.fill_diagonal(C, 1)
print(C)

# Print information related to the true probability of failure
e, v = np.linalg.eig(np.asarray(C))
print(e)
print(v)
beff = np.sqrt(np.max(e)) * beta
print(beff)
pf_true = stats.norm.cdf(-beta)
print(pf_true)

# %% md
#
# 3. Execute subset simulation with the :class:`.ModifiedMetropolisHastings` algorithm.

# %%

pf = np.zeros((ntrials, 1))
cov1 = np.zeros((ntrials, 1))
cov2 = np.zeros((ntrials, 1))
for i in range(ntrials):
    model = RunModel(model_script='local_pfn.py', model_object_name="run_python_model", ntasks=1, b_eff=beff, d=dim)
    dist = MultivariateNormal(mean=np.zeros((dim)), cov=C)
    x = dist.rvs(nsamples=1000, random_state=349857)
    sampling = ModifiedMetropolisHastings(dimension=dim, log_pdf_target=dist.log_pdf,
                                          n_chains=100, random_state=342985)

    x_ss = SubsetSimulation(sampling=sampling, runmodel_object=model, samples_init=x, conditional_probability=0.1,
                            nsamples_per_subset=1000)
    shutil.rmtree(model.model_dir)
    pf[i] = x_ss.failure_probability
    cov1[i] = x_ss.independent_chains_CoV
    cov2[i] = x_ss.dependent_chains_CoV

# %% md
#
# 4. Plot samples from each conditional level and print the subset simulation results.

# %%

for i in range(len(x_ss.performance_function_per_level)):
    plt.scatter(x_ss.samples[i][:, 0], x_ss.samples[i][:, 1], marker='o')
plt.show()

print('Mean Pf:', np.mean(pf))
pf[pf == 0] = 1e-100
print('Mean beta: ', -np.mean(stats.norm.ppf(pf)))
print('CoV: ', stats.variation(pf))
print('CoV beta: ', np.absolute(stats.variation(stats.norm.ppf(pf))))
print('CoV log10: ', np.absolute(stats.variation(np.log10(pf))))
print(np.mean(cov1))
print(np.mean(cov2))

# %% md
#
# 5. Plot histograms of the failure probabilities and the reliability indices from subset simulation.

# %%
plt.figure()
plt.hist(pf)
plt.show()

plt.figure()
beta = -stats.norm.ppf(pf)
plt.hist(beta)
plt.show()

# %% md
#
# 6. Execute subset simulation with the :class:`.Stretch` algorithm.

# %%

pf = np.zeros((ntrials, 1))
cov1 = np.zeros((ntrials, 1))
cov2 = np.zeros((ntrials, 1))
for i in range(ntrials):
    model = RunModel(model_script='local_pfn.py', model_object_name="run_python_model", ntasks=1, b_eff=beff, d=dim)
    dist = MultivariateNormal(mean=np.zeros((dim)), cov=C)
    x = dist.rvs(nsamples=1000, random_state=349857)
    sampling = Stretch(dimension=dim, log_pdf_target=dist.log_pdf, n_chains=100, random_state=342985)

    x_ss = SubsetSimulation(sampling=sampling, runmodel_object=model, samples_init=x, conditional_probability=0.1,
                            nsamples_per_subset=1000)
    shutil.rmtree(model.model_dir)
    pf[i] = x_ss.failure_probability
    cov1[i] = x_ss.independent_chains_CoV
    cov2[i] = x_ss.dependent_chains_CoV

# %% md
#
# 7. Plot samples from each conditional level and print the subset simulation results.

# %%

for i in range(len(x_ss.performance_function_per_level)):
    plt.scatter(x_ss.samples[i][:, 0], x_ss.samples[i][:, 1], marker='o')
plt.show()

print('Mean Pf:', np.mean(pf))
pf[pf == 0] = 1e-100
print('Mean beta: ', -np.mean(stats.norm.ppf(pf)))
print('CoV: ', stats.variation(pf))
print('CoV beta: ', np.absolute(stats.variation(stats.norm.ppf(pf))))
print('CoV log10: ', np.absolute(stats.variation(np.log10(pf))))
print(np.mean(cov1))
print(np.mean(cov2))

# %% md
#
# 8. Plot histograms of the failure probabilities and the reliability indices from subset simulation.

# %%
plt.figure()
plt.hist(pf)
plt.show()

plt.figure()
beta = -stats.norm.ppf(pf)
plt.hist(beta)
plt.show()
