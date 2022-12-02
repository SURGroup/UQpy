import numpy as np
from scipy.stats import multivariate_normal
from UQpy.distributions import Uniform, JointIndependent
from UQpy.sampling import MetropolisHastings, ParallelTemperingMCMC, SequentialTemperingMCMC


def log_rosenbrock(x):
    return -(100 * (x[:, 1] - x[:, 0] ** 2) ** 2 + (1 - x[:, 0]) ** 2) / 20


def log_intermediate(x, beta):
    return beta * log_rosenbrock(x)


def log_prior(x):
    loc, scale = -20., 40.
    return Uniform(loc=loc, scale=scale).log_pdf(x[:, 0]) + Uniform(loc=loc, scale=scale).log_pdf(x[:, 1])


def compute_potential(x, temper_param, log_intermediate_values):
    return log_intermediate_values / temper_param


random_state = np.random.RandomState(1234)
seed = -2. + 4. * random_state.rand(5, 2)
betas = [1. / np.sqrt(2.) ** i for i in range(10 - 1, -1, -1)]

prior_distribution = JointIndependent(marginals=[Uniform(loc=-2, scale=4), Uniform(loc=-2, scale=4)])


def test_parallel():
    samplers = [MetropolisHastings(burn_length=10, jump=2, seed=list(seed), dimension=2) for _ in range(len(betas))]
    mcmc = ParallelTemperingMCMC(log_pdf_intermediate=log_intermediate,
                                 distribution_reference=prior_distribution,
                                 n_iterations_between_sweeps=4,
                                 tempering_parameters=betas,
                                 random_state=3456,
                                 save_log_pdf=False, samplers=samplers)
    mcmc.run(nsamples_per_chain=100)
    assert mcmc.samples.shape == (500, 2)


def test_thermodynamic_integration():
    samplers = [MetropolisHastings(burn_length=10, jump=2, seed=list(seed), dimension=2) for _ in range(len(betas))]
    mcmc = ParallelTemperingMCMC(log_pdf_intermediate=log_intermediate,
                                 distribution_reference=prior_distribution,
                                 n_iterations_between_sweeps=4,
                                 tempering_parameters=betas,
                                 save_log_pdf=True,
                                 random_state=3456,
                                 samplers=samplers)
    mcmc.run(nsamples_per_chain=100)
    log_ev = mcmc.evaluate_normalization_constant(compute_potential=compute_potential, log_Z0=0.)
    assert np.round(log_ev, 4) == 0.203


def likelihood(x, b):
    mu1 = np.array([1., 1.])
    mu2 = -0.8 * np.ones(2)
    w1 = 0.5
    # Width of 0.1 in each dimension
    sigma1 = np.diag([0.02, 0.05])
    sigma2 = np.diag([0.05, 0.02])

    # Posterior is a mixture of two gaussians
    like = np.exp(np.logaddexp(np.log(w1) + multivariate_normal.logpdf(x=x, mean=mu1, cov=sigma1),
                               np.log(1. - w1) + multivariate_normal.logpdf(x=x, mean=mu2, cov=sigma2)))
    return like ** b


def test_sequential():
    prior = JointIndependent(marginals=[Uniform(loc=-2.0, scale=4.0), Uniform(loc=-2.0, scale=4.0)])
    sampler = MetropolisHastings(dimension=2, n_chains=20)
    test = SequentialTemperingMCMC(pdf_intermediate=likelihood,
                                   distribution_reference=prior,
                                   save_intermediate_samples=True,
                                   percentage_resampling=10,
                                   random_state=960242069,
                                   sampler=sampler,
                                   nsamples=100)
    assert np.round(test.evidence, 4) == 0.0656

