from UQpy.sampling.mcmc import *
import UQpy.distributions as Distributions


# Tests for parent MCMC and MH algorithm
def test_mh_1d_target_pdf():
    target = Distributions.Normal().pdf
    x = MetropolisHastings(dimension=1, pdf_target=target, chains_number=1, random_state=123, samples_number=10)
    assert round(float(x.samples[-1]), 3) == -1.291


def test_mh_1d_samples_per_chain():
    target = Distributions.Normal().pdf
    x = MetropolisHastings(dimension=1, pdf_target=target, chains_number=2, random_state=123,
                           samples_number_per_chain=5)
    assert round(float(x.samples[-1]), 3) == 0.474


def test_mh_1d_acceptance_rate():
    target = Distributions.Normal().pdf
    x = MetropolisHastings(dimension=1, pdf_target=target, chains_number=1, random_state=123, samples_number=100)
    assert round(float(x.acceptance_rate[0]), 3) == 0.707


def test_mh_1d_save_log_pdf():
    target = Distributions.Normal().pdf
    x = MetropolisHastings(dimension=1, pdf_target=target, chains_number=1, random_state=123, save_log_pdf=True,
                           samples_number=10)
    assert round(float(x.log_pdf_values[-1]), 3) == -1.752


def test_mh_1d_target_log_pdf():
    target = Distributions.Normal().log_pdf
    x = MetropolisHastings(dimension=1, log_pdf_target=target, chains_number=1, random_state=123, samples_number=10)
    assert round(float(x.samples[-1]), 3) == -1.291


def test_mh_2d():
    target = Distributions.MultivariateNormal([0., 0.]).pdf
    x = MetropolisHastings(dimension=2, pdf_target=target, chains_number=1, random_state=123, samples_number=10)
    assert [round(float(x.samples[-1][0]), 3), round(float(x.samples[-1][1]), 3)] == [-0.406, -1.217]


def test_mh_2d_burn_jump():
    target = Distributions.MultivariateNormal([0., 0.]).pdf
    x = MetropolisHastings(dimension=2, log_pdf_target=target, burn_length=10, jump=2, chains_number=1,
                           random_state=123, samples_number=10)
    assert x.iterations_number == 30


def test_mh_2d_nsamples_check():
    target = Distributions.MultivariateNormal([0., 0.]).pdf
    x = MetropolisHastings(dimension=2, pdf_target=target, chains_number=2, random_state=123, samples_number=60)
    assert x.samples_number_per_chain + x.samples_number == 90


def test_mh_2d_2chains():
    target = Distributions.MultivariateNormal([0., 0.]).pdf
    x = MetropolisHastings(dimension=2, pdf_target=target, chains_number=2, random_state=123, samples_number=60)
    assert [round(float(x.samples[-1][0]), 3), round(float(x.samples[-1][1]), 3)] == [-0.064, -0.533]


def test_mh_2d_2chains_non_concatenated():
    target = Distributions.MultivariateNormal([0., 0.]).pdf
    x = MetropolisHastings(dimension=2, pdf_target=target, concatenate_chains=False, chains_number=2, random_state=123,
                           samples_number=60)
    assert [[round(float(x.samples[-1][0][0]), 3), round(float(x.samples[-1][0][1]), 3)],
            [round(float(x.samples[-1][1][0]), 3), round(float(x.samples[-1][1][1]), 3)]] == [[1.767, 1.465],
                                                                                              [-0.064, -0.533]]


def test_mh_2d_seed():
    target = Distributions.MultivariateNormal([0., 0.]).pdf
    x = MetropolisHastings(pdf_target=target, seed=[0., 0.], chains_number=1, random_state=123, samples_number=10)
    assert [round(float(x.samples[-1][0]), 3), round(float(x.samples[-1][1]), 3)] == [-0.406, -1.217]


def test_mh_1d_symmetric_proposal_pdf():
    target = Distributions.Normal().pdf
    proposal = Distributions.Normal()
    x = MetropolisHastings(dimension=1, pdf_target=target, proposal=proposal, proposal_is_symmetric=True,
                           chains_number=1, random_state=123, samples_number=10)
    assert round(float(x.samples[-1]), 3) == -1.291


def test_mh_1d_asymmetric_proposal_pdf():
    target = Distributions.Normal().pdf
    proposal = Distributions.Normal()
    x = MetropolisHastings(dimension=1, pdf_target=target, proposal=proposal, proposal_is_symmetric=False,
                           chains_number=1, random_state=123, samples_number=10)
    assert round(float(x.samples[-1]), 3) == -1.291


def test_mmh_1d_burn_jump():
    target = Distributions.Normal().pdf
    x = ModifiedMetropolisHastings(dimension=1, pdf_target=target, burn_length=10,
                                   jump=2, chains_number=1, random_state=123, samples_number=10)
    assert round(float(x.samples[-1]), 3) == 0.497


def test_mmh_2d_list_target_pdf():
    target = [Distributions.Normal().pdf, Distributions.Normal().pdf]
    x = ModifiedMetropolisHastings(dimension=2, pdf_target=target, chains_number=1, random_state=123, samples_number=10)
    assert [round(float(x.samples[-1][0]), 3), round(float(x.samples[-1][1]), 3)] == [-0.810, 0.173]


def test_mmh_2d_list_target_log_pdf():
    target = [Distributions.Normal().log_pdf, Distributions.Normal().log_pdf]
    x = ModifiedMetropolisHastings(dimension=2, log_pdf_target=target, chains_number=1, random_state=123,
                                   samples_number=10)
    assert [round(float(x.samples[-1][0]), 3), round(float(x.samples[-1][1]), 3)] == [-0.810, 0.173]


def test_mmh_2d_joint_proposal():
    target = Distributions.MultivariateNormal([0., 0.]).pdf
    proposal = Distributions.JointIndependent(marginals=[Distributions.Normal(scale=0.2),
                                                         Distributions.Normal(scale=0.2)])
    x = ModifiedMetropolisHastings(dimension=2, pdf_target=target, chains_number=1, proposal=proposal, random_state=123,
                                   samples_number=10)
    assert [round(float(x.samples[-1][0]), 3), round(float(x.samples[-1][1]), 3)] == [-0.783, -0.195]


def test_mmh_2d_list_proposal():
    target = Distributions.MultivariateNormal([0., 0.]).pdf
    proposal = [Distributions.Normal(scale=0.2), Distributions.Normal(scale=0.2)]
    x = ModifiedMetropolisHastings(dimension=2, pdf_target=target, chains_number=1, proposal=proposal, random_state=123,
                                   samples_number=10)
    assert [round(float(x.samples[-1][0]), 3), round(float(x.samples[-1][1]), 3)] == [-0.783, -0.195]


def test_mmh_2d_single1d_proposal():
    target = Distributions.MultivariateNormal([0., 0.]).pdf
    proposal = Distributions.Normal(scale=0.2)
    x = ModifiedMetropolisHastings(dimension=2, pdf_target=target, chains_number=1, proposal=proposal, random_state=123,
                                   samples_number=10)
    assert [round(float(x.samples[-1][0]), 3), round(float(x.samples[-1][1]), 3)] == [-0.783, -0.195]


def test_mmh_2d_list_proposal_log_target():
    target = [Distributions.Normal().log_pdf, Distributions.Normal().log_pdf]
    proposal = [Distributions.Normal(scale=0.2), Distributions.Normal(scale=0.2)]
    x = ModifiedMetropolisHastings(dimension=2, log_pdf_target=target, chains_number=1, proposal=proposal,
                                   random_state=123, samples_number=10)
    assert [round(float(x.samples[-1][0]), 3), round(float(x.samples[-1][1]), 3)] == [-0.783, -0.195]


def test_dram_1d_burn_jump():
    target = Distributions.Normal().pdf
    x = DRAM(dimension=1, pdf_target=target, burn_length=10, jump=2, chains_number=1, random_state=123,
             samples_number=10)
    assert round(float(x.samples[-1]), 3) == 0.935


def test_dream_1d_burn_jump():
    target = Distributions.Normal().pdf
    x = DREAM(pdf_target=target, burn_length=10, jump=2, dimension=1, chains_number=10, random_state=123,
              samples_number=20)
    assert round(float(x.samples[-1]), 3) == 0.0


def test_dream_1d_check_chains():
    target = Distributions.Normal().pdf
    x = DREAM(pdf_target=target, burn_length=0, jump=2, save_log_pdf=True, dimension=1, check_chains=(1000, 1),
              chains_number=20, random_state=123, samples_number=2000)
    assert (round(float(x.samples[-1]), 3) == 0.593)


def test_dream_1d_adapt_chains():
    target = Distributions.Normal().pdf
    x = DREAM(pdf_target=target, burn_length=1000, jump=2, save_log_pdf=True, dimension=1,
              crossover_adaptation=(1000, 1), chains_number=20, random_state=123, samples_number=2000)
    assert (round(float(x.samples[-1]), 3) == -0.446)


def test_stretch_1d_burn_jump():
    target = Distributions.Normal().pdf
    x = Stretch(pdf_target=target, burn_length=10, jump=2, dimension=1, chains_number=2, random_state=123,
                samples_number=10)
    assert round(float(x.samples[-1]), 3) == -0.961


def test_unconcatenate_chains_mcmc():
    target = Distributions.Normal().pdf
    x = ModifiedMetropolisHastings(dimension=1, pdf_target=target, burn_length=10, jump=2, chains_number=2,
                                   save_log_pdf=True, random_state=123)
    x.run(samples_number=5)
    x.run(samples_number=5)
    assert (round(float(x.samples[-1]), 3) == -0.744)
