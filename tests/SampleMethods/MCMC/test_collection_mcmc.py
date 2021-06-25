from UQpy.SampleMethods.MCMC import *
import UQpy.Distributions as Distributions


# Tests for parent MCMC and MH algorithm
def test_mh_1d_target_pdf():
    target = Distributions.Normal().pdf
    x = MH(dimension=1, pdf_target=target, nsamples=10, nchains=1, random_state=123)
    assert round(float(x.samples[-1]), 3) == -1.291


def test_mh_1d_samplesperchain():
    target = Distributions.Normal().pdf
    x = MH(dimension=1, pdf_target=target, nsamples_per_chain=5, nchains=2, random_state=123)
    assert round(float(x.samples[-1]), 3) == 0.474


def test_mh_1d_acceptancerate():
    target = Distributions.Normal().pdf
    x = MH(dimension=1, pdf_target=target, nsamples=100, nchains=1, random_state=123)
    assert round(float(x.acceptance_rate[0]), 3) == 0.707


def test_mh_1d_savelogpdf():
    target = Distributions.Normal().pdf
    x = MH(dimension=1, pdf_target=target, nsamples=10, nchains=1, save_log_pdf=True, random_state=123)
    assert round(float(x.log_pdf_values[-1]), 3) == -1.752


def test_mh_1d_target_logpdf():
    target = Distributions.Normal().log_pdf
    x = MH(dimension=1, log_pdf_target=target, nsamples=10, nchains=1, random_state=123)
    assert round(float(x.samples[-1]), 3) == -1.291


def test_mh_2d():
    target = Distributions.MVNormal([0., 0.]).pdf
    x = MH(dimension=2, pdf_target=target, nsamples=10, nchains=1, random_state=123)
    assert [round(float(x.samples[-1][0]), 3), round(float(x.samples[-1][1]), 3)] == [-0.406, -1.217]


def test_mh_2d_burnjump():
    target = Distributions.MVNormal([0., 0.]).pdf
    x = MH(dimension=2, pdf_target=target, nburn=10, jump=2, nsamples=10, nchains=1, random_state=123)
    # y = MH(dimension=2, pdf_target=target, nsamples=31, nchains=1, random_state=123)
    assert x.niterations == 30


def test_mh_2d_nsamplecheck():
    target = Distributions.MVNormal([0., 0.]).pdf
    x = MH(dimension=2, pdf_target=target, nsamples=60, nchains=2, random_state=123)
    assert x.nsamples_per_chain + x.nsamples == 90


def test_mh_2d_2chainz():
    target = Distributions.MVNormal([0., 0.]).pdf
    x = MH(dimension=2, pdf_target=target, nsamples=60, nchains=2, random_state=123)
    assert [round(float(x.samples[-1][0]), 3), round(float(x.samples[-1][1]), 3)] == [-0.064, -0.533]


def test_mh_2d_2chainz_nonconcatenated():
    target = Distributions.MVNormal([0., 0.]).pdf
    x = MH(dimension=2, pdf_target=target, nsamples=60, nchains=2, concat_chains=False, random_state=123)
    assert [[round(float(x.samples[-1][0][0]), 3), round(float(x.samples[-1][0][1]), 3)], [round(float(x.samples[-1][1][0]), 3), round(float(x.samples[-1][1][1]), 3)]] == [[1.767, 1.465], [-0.064, -0.533]]


def test_mh_2d_seed():
    target = Distributions.MVNormal([0., 0.]).pdf
    x = MH(pdf_target=target, nsamples=10, nchains=1, seed=[0., 0.], random_state=123)
    assert [round(float(x.samples[-1][0]), 3), round(float(x.samples[-1][1]), 3)] == [-0.406, -1.217]


def test_mh_1d_symmetric_proposal_pdf():
    target = Distributions.Normal().pdf
    proposal = Distributions.Normal()
    x = MH(dimension=1, pdf_target=target, nsamples=10, nchains=1, proposal=proposal, proposal_is_symmetric=True,
           random_state=123)
    assert round(float(x.samples[-1]), 3) == -1.291


def test_mh_1d_asymmetric_proposal_pdf():
    target = Distributions.Normal().pdf
    proposal = Distributions.Normal()
    x = MH(dimension=1, pdf_target=target, nsamples=10, nchains=1, proposal=proposal, proposal_is_symmetric=False,
           random_state=123)
    assert round(float(x.samples[-1]), 3) == -1.291


def test_mmh_1d_burnjump():
    target = Distributions.Normal().pdf
    x = MMH(dimension=1, pdf_target=target, nburn=10, jump=2, nsamples=10, nchains=1, random_state=123)
    assert round(float(x.samples[-1]), 3) == 0.497


def test_mmh_2d_list_target_pdf():
    target = [Distributions.Normal().pdf, Distributions.Normal().pdf]
    x = MMH(dimension=2, pdf_target=target, nsamples=10, nchains=1, random_state=123)
    assert [round(float(x.samples[-1][0]), 3), round(float(x.samples[-1][1]), 3)] == [-0.810, 0.173]


def test_mmh_2d_list_target_logpdf():
    target = [Distributions.Normal().log_pdf, Distributions.Normal().log_pdf]
    x = MMH(dimension=2, log_pdf_target=target, nsamples=10, nchains=1, random_state=123)
    assert [round(float(x.samples[-1][0]), 3), round(float(x.samples[-1][1]), 3)] == [-0.810, 0.173]


def test_mmh_2d_joint_proposal():
    target = Distributions.MVNormal([0., 0.]).pdf
    proposal = Distributions.JointInd(marginals=[Distributions.Normal(scale=0.2), Distributions.Normal(scale=0.2)])
    x = MMH(dimension=2, pdf_target=target, nsamples=10, nchains=1, proposal=proposal, random_state=123)
    assert [round(float(x.samples[-1][0]), 3), round(float(x.samples[-1][1]), 3)] == [-0.783, -0.195]


def test_mmh_2d_list_proposal():
    target = Distributions.MVNormal([0., 0.]).pdf
    proposal = [Distributions.Normal(scale=0.2), Distributions.Normal(scale=0.2)]
    x = MMH(dimension=2, pdf_target=target, nsamples=10, nchains=1, proposal=proposal, random_state=123)
    assert [round(float(x.samples[-1][0]), 3), round(float(x.samples[-1][1]), 3)] == [-0.783, -0.195]


def test_mmh_2d_single1d_proposal():
    target = Distributions.MVNormal([0., 0.]).pdf
    proposal = Distributions.Normal(scale=0.2)
    x = MMH(dimension=2, pdf_target=target, nsamples=10, nchains=1, proposal=proposal, random_state=123)
    assert [round(float(x.samples[-1][0]), 3), round(float(x.samples[-1][1]), 3)] == [-0.783, -0.195]


def test_mmh_2d_list_proposal_log_target():
    target = [Distributions.Normal().log_pdf, Distributions.Normal().log_pdf]
    proposal = [Distributions.Normal(scale=0.2), Distributions.Normal(scale=0.2)]
    x = MMH(dimension=2, log_pdf_target=target, nsamples=10, nchains=1, proposal=proposal, random_state=123)
    assert [round(float(x.samples[-1][0]), 3), round(float(x.samples[-1][1]), 3)] == [-0.783, -0.195]


def test_dram_1d_burnjump():
    target = Distributions.Normal().pdf
    x = DRAM(dimension=1, pdf_target=target, nburn=10, jump=2, nsamples=10, nchains=1, random_state=123)
    assert round(float(x.samples[-1]), 3) == 0.935


# def test_dram_savecov():
#     target = Distributions.MVNormal([0., 0.]).pdf
#     x = DRAM(dimension=2, pdf_target=target, nburn=10, jump=2, nsamples=10, nchains=1, save_covariance=True,
#              random_state=123)
#     assert round(float(x.adaptive_covariance[-1][0][0][0]), 3) == 1.0


def test_dream_1d_burnjump():
    target = Distributions.Normal().pdf
    x = DREAM(pdf_target=target, nburn=10, jump=2, dimension=1, nsamples=20, nchains=10, random_state=123)
    assert round(float(x.samples[-1]), 3) == 0.0


def test_dream_1d_checkchains():
    target = Distributions.Normal().pdf
    x = DREAM(pdf_target=target, nburn=0, jump=2, dimension=1, save_log_pdf=True, nsamples=2000,
              check_chains=(1000, 1), nchains=20, random_state=123)
    assert (round(float(x.samples[-1]), 3) == 0.593)


def test_dream_1d_adaptchains():
    target = Distributions.Normal().pdf
    x = DREAM(pdf_target=target, nburn=1000, jump=2, dimension=1, save_log_pdf=True, nsamples=2000,
              adapt_cr=(1000, 1), nchains=20, random_state=123)
    assert (round(float(x.samples[-1]), 3) == -0.446)


def test_stretch_1d_burnjump():
    target = Distributions.Normal().pdf
    x = Stretch(pdf_target=target, nburn=10, jump=2, dimension=1, nsamples=10, nchains=2, random_state=123)
    assert round(float(x.samples[-1]), 3) == -0.961


def test_unconcat_chains_mcmc():
    target = Distributions.Normal().pdf
    x = MMH(dimension=1, pdf_target=target, nburn=10, jump=2, nchains=2, save_log_pdf=True, random_state=123)
    x.run(nsamples=5)
    x.run(nsamples=5)
    assert (round(float(x.samples[-1]), 3) == -0.744)
