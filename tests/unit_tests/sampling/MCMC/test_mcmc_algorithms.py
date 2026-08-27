from UQpy.sampling.mcmc import *
import UQpy.distributions as Distributions
import numpy as np


# Tests for parent MCMC and MH algorithm
def test_mh_1d_target_pdf():
    target = Distributions.Normal().pdf
    x = MetropolisHastings(
        dimension=1, pdf_target=target, n_chains=1, random_state=123, nsamples=10
    )
    np.testing.assert_allclose(x.samples[-1], -1.291, atol=1e-3)


def test_mh_1d_samples_per_chain():
    target = Distributions.Normal().pdf
    x = MetropolisHastings(
        dimension=1,
        pdf_target=target,
        n_chains=2,
        random_state=123,
        nsamples_per_chain=5,
    )
    np.testing.assert_allclose(x.samples[-1], 0.474, atol=1e-3)


def test_mh_1d_acceptance_rate():
    target = Distributions.Normal().pdf
    x = MetropolisHastings(
        dimension=1, pdf_target=target, n_chains=1, random_state=123, nsamples=100
    )
    np.testing.assert_allclose(x.acceptance_rate[0], 0.707, atol=1e-3)


def test_mh_1d_save_log_pdf():
    target = Distributions.Normal().pdf
    x = MetropolisHastings(
        dimension=1,
        pdf_target=target,
        n_chains=1,
        random_state=123,
        save_log_pdf=True,
        nsamples=10,
    )
    np.testing.assert_allclose(x.log_pdf_values[-1], -1.752, atol=1e-3)


def test_mh_1d_target_log_pdf():
    target = Distributions.Normal().log_pdf
    x = MetropolisHastings(
        dimension=1, log_pdf_target=target, n_chains=1, random_state=123, nsamples=10
    )
    np.testing.assert_allclose(x.samples[-1], -1.291, atol=1e-3)


def test_mh_2d():
    target = Distributions.MultivariateNormal([0.0, 0.0]).pdf
    x = MetropolisHastings(
        dimension=2, pdf_target=target, n_chains=1, random_state=123, nsamples=10
    )
    np.testing.assert_allclose(x.samples[-1], [-0.406, -1.217], atol=1e-3)


def test_mh_2d_burn_jump():
    target = Distributions.MultivariateNormal([0.0, 0.0]).pdf
    x = MetropolisHastings(
        dimension=2,
        log_pdf_target=target,
        burn_length=10,
        jump=2,
        n_chains=1,
        random_state=123,
        nsamples=10,
    )
    assert x.iterations_number == 30


def test_mh_2d_nsamples_check():
    target = Distributions.MultivariateNormal([0.0, 0.0]).pdf
    x = MetropolisHastings(
        dimension=2, pdf_target=target, n_chains=2, random_state=123, nsamples=60
    )
    assert x.nsamples_per_chain + x.samples_counter == 90


def test_mh_2d_2chains():
    target = Distributions.MultivariateNormal([0.0, 0.0]).pdf
    x = MetropolisHastings(
        dimension=2, pdf_target=target, n_chains=2, random_state=123, nsamples=60
    )
    np.testing.assert_allclose(x.samples[-1], [-0.064, -0.533], atol=1e-3)


def test_mh_2d_2chains_non_concatenated():
    target = Distributions.MultivariateNormal([0.0, 0.0]).pdf
    x = MetropolisHastings(
        dimension=2,
        pdf_target=target,
        concatenate_chains=False,
        n_chains=2,
        random_state=123,
        nsamples=60,
    )
    np.testing.assert_allclose(
        x.samples[-1], [[1.767, 1.465], [-0.064, -0.533]], atol=1e-3
    )


def test_mh_2d_seed():
    target = Distributions.MultivariateNormal([0.0, 0.0]).pdf
    x = MetropolisHastings(
        pdf_target=target, seed=[0.0, 0.0], n_chains=1, random_state=123, nsamples=10
    )
    np.testing.assert_allclose(x.samples[-1], [-0.406, -1.217], atol=1e-3)


def test_mh_1d_symmetric_proposal_pdf():
    target = Distributions.Normal().pdf
    proposal = Distributions.Normal()
    x = MetropolisHastings(
        dimension=1,
        pdf_target=target,
        proposal=proposal,
        proposal_is_symmetric=True,
        n_chains=1,
        random_state=123,
        nsamples=10,
    )
    np.testing.assert_allclose(x.samples[-1], -1.291, atol=1e-3)


def test_mh_1d_asymmetric_proposal_pdf():
    target = Distributions.Normal().pdf
    proposal = Distributions.Normal()
    x = MetropolisHastings(
        dimension=1,
        pdf_target=target,
        proposal=proposal,
        proposal_is_symmetric=False,
        n_chains=1,
        random_state=123,
        nsamples=10,
    )
    np.testing.assert_allclose(x.samples[-1], -1.291, atol=1e-3)


def test_mmh_1d_burn_jump():
    target = Distributions.Normal().pdf
    x = ModifiedMetropolisHastings(
        dimension=1,
        pdf_target=target,
        burn_length=10,
        jump=2,
        n_chains=1,
        random_state=123,
        nsamples=10,
    )
    np.testing.assert_allclose(x.samples[-1], 0.497, atol=1e-3)


def test_mmh_2d_list_target_pdf():
    target = [Distributions.Normal().pdf, Distributions.Normal().pdf]
    x = ModifiedMetropolisHastings(
        dimension=2, pdf_target=target, n_chains=1, random_state=123, nsamples=10
    )
    np.testing.assert_allclose(x.samples[-1], [-0.810, 0.173], atol=1e-3)


def test_mmh_2d_list_target_log_pdf():
    target = [Distributions.Normal().log_pdf, Distributions.Normal().log_pdf]
    x = ModifiedMetropolisHastings(
        dimension=2, log_pdf_target=target, n_chains=1, random_state=123, nsamples=10
    )
    np.testing.assert_allclose(x.samples[-1], [-0.810, 0.173], atol=1e-3)


def test_mmh_2d_joint_proposal():
    target = Distributions.MultivariateNormal([0.0, 0.0]).pdf
    proposal = Distributions.JointIndependent(
        marginals=[Distributions.Normal(scale=0.2), Distributions.Normal(scale=0.2)]
    )
    x = ModifiedMetropolisHastings(
        dimension=2,
        pdf_target=target,
        n_chains=1,
        proposal=proposal,
        random_state=123,
        nsamples=10,
    )
    np.testing.assert_allclose(x.samples[-1], [-0.783, -0.195], atol=1e-3)


def test_mmh_2d_list_proposal():
    target = Distributions.MultivariateNormal([0.0, 0.0]).pdf
    proposal = [Distributions.Normal(scale=0.2), Distributions.Normal(scale=0.2)]
    x = ModifiedMetropolisHastings(
        dimension=2,
        pdf_target=target,
        n_chains=1,
        proposal=proposal,
        random_state=123,
        nsamples=10,
    )
    np.testing.assert_allclose(x.samples[-1], [-0.783, -0.195], atol=1e-3)


def test_mmh_2d_single1d_proposal():
    target = Distributions.MultivariateNormal([0.0, 0.0]).pdf
    proposal = Distributions.Normal(scale=0.2)
    x = ModifiedMetropolisHastings(
        dimension=2,
        pdf_target=target,
        n_chains=1,
        proposal=proposal,
        random_state=123,
        nsamples=10,
    )
    np.testing.assert_allclose(x.samples[-1], [-0.783, -0.195], atol=1e-3)


def test_mmh_2d_list_proposal_log_target():
    target = [Distributions.Normal().log_pdf, Distributions.Normal().log_pdf]
    proposal = [Distributions.Normal(scale=0.2), Distributions.Normal(scale=0.2)]
    x = ModifiedMetropolisHastings(
        dimension=2,
        log_pdf_target=target,
        n_chains=1,
        proposal=proposal,
        random_state=123,
        nsamples=10,
    )
    np.testing.assert_allclose(x.samples[-1], [-0.783, -0.195], atol=1e-3)


def test_dram_1d_burn_jump():
    target = Distributions.Normal().pdf
    x = DRAM(
        dimension=1,
        pdf_target=target,
        burn_length=10,
        jump=2,
        n_chains=1,
        random_state=123,
        nsamples=10,
    )
    np.testing.assert_allclose(x.samples[-1], 0.935, atol=1e-3)


def test_dream_1d_burn_jump():
    target = Distributions.Normal().pdf
    x = DREAM(
        pdf_target=target,
        burn_length=10,
        jump=2,
        dimension=1,
        n_chains=10,
        random_state=123,
        nsamples=20,
    )
    np.testing.assert_allclose(x.samples[-1], 0.0, atol=1e-3)


def test_dream_1d_check_chains():
    target = Distributions.Normal().pdf
    x = DREAM(
        pdf_target=target,
        burn_length=0,
        jump=2,
        save_log_pdf=True,
        dimension=1,
        check_chains=(1000, 1),
        n_chains=20,
        random_state=123,
        nsamples=100000,
    )

    samples_flat = x.samples.flatten()
    sample_mean = np.mean(samples_flat)
    sample_std = np.std(samples_flat, ddof=1)

    np.testing.assert_allclose(sample_mean, 0.0, atol=1e-1)
    np.testing.assert_allclose(sample_std, 1.0, atol=1e-1)


def test_dream_1d_adapt_chains():
    target = Distributions.Normal().pdf
    x = DREAM(
        pdf_target=target,
        burn_length=1000,
        jump=2,
        save_log_pdf=True,
        dimension=1,
        crossover_adaptation=(1000, 1),
        n_chains=20,
        random_state=123,
        nsamples=100000,
    )
    # Test that samples with crossover adaptation still converge to N(0,1)
    samples_flat = x.samples.flatten()
    sample_mean = np.mean(samples_flat)
    sample_std = np.std(samples_flat, ddof=1)

    np.testing.assert_allclose(sample_mean, 0.0, atol=1e-1)
    np.testing.assert_allclose(sample_std, 1.0, atol=1e-1)


def test_stretch_1d_burn_jump():
    target = Distributions.Normal().pdf
    x = Stretch(
        pdf_target=target,
        burn_length=10,
        jump=2,
        dimension=1,
        n_chains=2,
        random_state=123,
        nsamples=10,
    )
    np.testing.assert_allclose(x.samples[-1], -0.961, atol=1e-3)


def test_unconcatenate_chains_mcmc():
    target = Distributions.Normal().pdf
    x = ModifiedMetropolisHastings(
        dimension=1,
        pdf_target=target,
        burn_length=10,
        jump=2,
        n_chains=2,
        save_log_pdf=True,
        random_state=123,
    )
    x.run(nsamples=5)
    x.run(nsamples=5)
    np.testing.assert_allclose(x.samples[-1], -0.744, atol=1e-3)
