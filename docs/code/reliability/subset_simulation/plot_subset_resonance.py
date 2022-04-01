"""

Subset Simulation for a SDOF resonance problem
======================================================================

Here, the Modified Metropolis Hastings Algorithm for MCMC is compared to the Affine Invariant Ensemble sampler for
MCMC in subset simulation to estimate probability of failure of a single degree of freedom system subjected to
harmonic excitation with a given frequency.

Problem Definition
-------------------
A stochastic single degree of freedom system having stiffness :math:`k\sim N(\mu_k,\sigma_k)` and mass
:math:`m\sim N(\mu_m,\sigma_m)` is excited by a sinusoidal load with frequency :math:`\omega \ rad/sec`.
The system is undamped and has equation of motion given by:

.. math:: `m\ddot{u}+ku=sin(\omega t)`

Resonance occurs when the natural frequency of the system :math:`\omega_n=\sqrt{\dfrac{k}{m}}=\omega \ rad/sec`.
To avoid resonance, we consider failure of the system to be associated with the natural frequency being within a
threshold, :math:`\epsilon` of the excitation frequency $\omega$. That is, failure of the system occurs when
:math:`\omega-\epsilon\le\sqrt{\dfrac{k}{m}}\le\omega+\epsilon`.

"""

# %% md
#
# 1. Import the necessary libraries

# %%
import shutil

from UQpy.reliability import SubsetSimulation
from UQpy.run_model.RunModel import RunModel
from UQpy.sampling import Stretch, ModifiedMetropolisHastings, MonteCarloSampling
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from UQpy.distributions import Normal, JointIndependent, MultivariateNormal

# %% md
#
# Set Parameters
#
# :math:`\omega=5`
#
# :math:`\epsilon = 0.0001`
#
# :math:`\mu_k=125`
#
# :math:`\sigma_k=20`
#
# :math:`\mu_m=5`
#
# :math:`\sigma_m=1`

# %%

omega = 6
epsilon = 0.0001
mu_m = 5
sigma_m = 1
mu_k = 125
sigma_k = 20
m = np.linspace(mu_m - 3 * sigma_m, mu_m + 3 * sigma_m, 101)
k_hi = (omega - epsilon) ** 2 * m
k_lo = (omega + epsilon) ** 2 * m

# %% md
#
# Plot the failure domain

# %%

x = np.linspace(2, 8, 1000)
y = np.linspace(25, 225, 1000)

X, Y = np.meshgrid(x, y)
Z = np.zeros((1000, 1000))

d1 = Normal(loc=5, scale=1)
d2 = Normal(loc=125, scale=20)

dist = JointIndependent(marginals=[d1, d2])

for i in range(len(x)):
    Z[i, :] = dist.pdf(np.append(np.atleast_2d(X[i, :]), np.atleast_2d(Y[i, :]), 0).T)

fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z, 15)
plt.plot(m, k_hi, 'k')
plt.plot(m, k_lo, 'k')
# plt.fill_between(m,k_lo,k_hi)
plt.xlim([mu_m - 3 * sigma_m, mu_m + 3 * sigma_m])
plt.ylim([mu_k - 3 * sigma_k, mu_k + 3 * sigma_k])
plt.xlabel(r'Mass ($m$)')
plt.ylabel(r'Stiffness ($k$)')
plt.grid(True)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z, 15)
plt.plot(m, k_hi, 'k')
plt.plot(m, k_lo, 'k')
# plt.fill_between(m,k_lo,k_hi)
plt.xlim([3.5, 4.5])
plt.ylim([130, 150])
plt.xlabel(r'Mass ($m$)')
plt.ylabel(r'Stiffness ($k$)')
plt.grid(True)
plt.tight_layout()
plt.show()

model = RunModel(model_script='local_Resonance_pfn.py', model_object_name="RunPythonModel", ntasks=1)

# %% md
#
# Monte Carlo Simulation

# %%

x_mcs = MonteCarloSampling(distributions=[d1, d2])
x_mcs.run(nsamples=1000000)

model.run(samples=x_mcs.samples)

shutil.rmtree(model.model_dir)
A = np.asarray(model.qoi_list) < 0
pf = np.shape(np.asarray(model.qoi_list)[np.asarray(model.qoi_list) < 0])[0] / 1000000
print(pf)

ntrials = 1
pf_stretch = np.zeros((ntrials, 1))
cov1_stretch = np.zeros((ntrials, 1))
cov2_stretch = np.zeros((ntrials, 1))
m = np.ones(2)
m[0] = 5
m[1] = 125
C = np.eye(2)
C[0, 0] = 1
C[1, 1] = 20 ** 2

for i in range(ntrials):
    model = RunModel(model_script='local_Resonance_pfn.py', model_object_name="RunPythonModel", ntasks=1)
    dist = MultivariateNormal(mean=m, cov=C)
    xx = dist.rvs(nsamples=1000, random_state=123)
    xx1 = dist.rvs(nsamples=100, random_state=123)

    sampling=Stretch(dimension=2, n_chains=100, log_pdf_target=dist.log_pdf)
    x_ss_stretch = SubsetSimulation(sampling=sampling, runmodel_object=model, conditional_probability=0.1,
                                    nsamples_per_subset=1000, samples_init=xx, )
    shutil.rmtree(model.model_dir)
    pf_stretch[i] = x_ss_stretch.failure_probability
    cov1_stretch[i] = x_ss_stretch.independent_chains_CoV
    cov2_stretch[i] = x_ss_stretch.dependent_chains_CoV

print(pf_stretch)
print(np.mean(pf_stretch, 0))
b_stretch = -stats.norm.ppf(pf_stretch)
print(np.mean(b_stretch, 0))
print(stats.variation(b_stretch))


for i in range(len(x_ss_stretch.performance_function_per_level)):
    plt.scatter(x_ss_stretch.samples[i][:, 0], x_ss_stretch.samples[i][:, 1], marker='o')

plt.xlim([mu_m - 3 * sigma_m, mu_m + 3 * sigma_m])
plt.ylim([mu_k - 3 * sigma_k, mu_k + 3 * sigma_k])
plt.xlabel(r'Mass ($m$)')
plt.ylabel(r'Stiffness ($k$)')
plt.grid(True)
plt.tight_layout()
plt.show()


ntrials = 100
pf_mmh = np.zeros((ntrials, 1))
cov1_mmh = np.zeros((ntrials, 1))
cov2_mmh = np.zeros((ntrials, 1))
m = np.ones(2)
m[0] = 5
m[1] = 125
C = np.eye(2)
C[0, 0] = 1
C[1, 1] = 20 ** 2

for i in range(ntrials):
    model = RunModel(model_script='local_Resonance_pfn.py', model_object_name="RunPythonModel", ntasks=1)
    dist = MultivariateNormal(mean=m, cov=C)
    xx = dist.rvs(nsamples=1000, random_state=123)
    xx1 = dist.rvs(nsamples=100, random_state=123)

    sampling = ModifiedMetropolisHastings(dimension=2, n_chains=100, log_pdf_target=dist.log_pdf)
    x_ss_mmh = SubsetSimulation(sampling=sampling, runmodel_object=model, conditional_probability=0.1,
                                nsamples_per_subset=1000, samples_init=xx)

    shutil.rmtree(model.model_dir)
    pf_mmh[i] = x_ss_mmh.failure_probability
    cov1_mmh[i] = x_ss_mmh.independent_chains_CoV
    cov2_mmh[i] = x_ss_mmh.dependent_chains_CoV

pf_mmh[pf_mmh == 0] = 1e-100
print(np.mean(pf_mmh, 0))
b_mmh = -stats.norm.ppf(pf_mmh)
print(np.mean(b_mmh, 0))
print(stats.variation(b_mmh))


for i in range(len(x_ss_mmh.performance_function_per_level)):
    plt.scatter(x_ss_mmh.samples[i][:, 0], x_ss_mmh.samples[i][:, 1], marker='o')

plt.xlim([mu_m - 3 * sigma_m, mu_m + 3 * sigma_m])
plt.ylim([mu_k - 3 * sigma_k, mu_k + 3 * sigma_k])
plt.xlabel(r'Mass ($m$)')
plt.ylabel(r'Stiffness ($k$)')
plt.grid(True)
plt.tight_layout()
plt.show()