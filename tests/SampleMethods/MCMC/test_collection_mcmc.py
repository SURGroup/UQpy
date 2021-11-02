from UQpy.sampling.mcmc import *
import UQpy.distributions as Distributions
from UQpy.sampling.input_data.MhInput import *
from UQpy.sampling.input_data.MmhInput import *
from UQpy.sampling.input_data.DramInput import *
from UQpy.sampling.input_data.DreamInput import *
from UQpy.sampling.input_data.StretchInput import *

# Tests for parent MCMC and MH algorithm
def test_mh_1d_target_pdf():
    target = Distributions.Normal().pdf
    mh_input = MhInput(dimension=1, pdf_target=target,
                       chains_number=1, random_state=123)
    x = MetropolisHastings(mh_input=mh_input, samples_number=10)
    assert round(float(x.samples[-1]), 3) == -1.291


def test_mh_1d_samplesperchain():
    target = Distributions.Normal().pdf
    mh_input = MhInput(dimension=1, pdf_target=target,
                       chains_number=2, random_state=123)
    x = MetropolisHastings(mh_input=mh_input, samples_number_per_chain=5)
    assert round(float(x.samples[-1]), 3) == 0.474


def test_mh_1d_acceptancerate():
    target = Distributions.Normal().pdf
    mh_input = MhInput(dimension=1, pdf_target=target,
                       chains_number=1, random_state=123)
    x = MetropolisHastings(mh_input=mh_input, samples_number=100)
    assert round(float(x.acceptance_rate[0]), 3) == 0.707


def test_mh_1d_savelogpdf():
    target = Distributions.Normal().pdf
    mh_input = MhInput(dimension=1, pdf_target=target,
                       chains_number=1, random_state=123, save_log_pdf=True)
    x = MetropolisHastings(mh_input=mh_input, samples_number=10)
    assert round(float(x.log_pdf_values[-1]), 3) == -1.752


def test_mh_1d_target_logpdf():
    target = Distributions.Normal().log_pdf
    mh_input = MhInput(dimension=1, log_pdf_target=target,
                       chains_number=1, random_state=123)
    x = MetropolisHastings(mh_input=mh_input, samples_number=10)
    assert round(float(x.samples[-1]), 3) == -1.291


def test_mh_2d():
    target = Distributions.MultivariateNormal([0., 0.]).pdf
    mh_input = MhInput(dimension=2, pdf_target=target,
                       chains_number=1, random_state=123)
    x = MetropolisHastings(mh_input=mh_input, samples_number=10)
    assert [round(float(x.samples[-1][0]), 3), round(float(x.samples[-1][1]), 3)] == [-0.406, -1.217]


def test_mh_2d_burnjump():
    target = Distributions.MultivariateNormal([0., 0.]).pdf
    mh_input = MhInput(dimension=2, log_pdf_target=target, burn_length=10,
                       jump=2, chains_number=1, random_state=123)
    x = MetropolisHastings(mh_input=mh_input, samples_number=10)
    assert x.iterations_number == 30


def test_mh_2d_nsamplecheck():
    target = Distributions.MultivariateNormal([0., 0.]).pdf
    mh_input = MhInput(dimension=2, pdf_target=target,
                       chains_number=2, random_state=123)
    x = MetropolisHastings(mh_input=mh_input, samples_number=60)
    assert x.samples_number_per_chain + x.samples_number == 90


def test_mh_2d_2chainz():
    target = Distributions.MultivariateNormal([0., 0.]).pdf
    mh_input = MhInput(dimension=2, pdf_target=target,
                       chains_number=2, random_state=123)
    x = MetropolisHastings(mh_input=mh_input, samples_number=60)
    assert [round(float(x.samples[-1][0]), 3), round(float(x.samples[-1][1]), 3)] == [-0.064, -0.533]


def test_mh_2d_2chainz_nonconcatenated():
    target = Distributions.MultivariateNormal([0., 0.]).pdf
    mh_input = MhInput(dimension=2, pdf_target=target, concatenate_chains=False,
                       chains_number=2, random_state=123)
    x = MetropolisHastings(mh_input=mh_input, samples_number=60)
    assert [[round(float(x.samples[-1][0][0]), 3), round(float(x.samples[-1][0][1]), 3)], [round(float(x.samples[-1][1][0]), 3), round(float(x.samples[-1][1][1]), 3)]] == [[1.767, 1.465], [-0.064, -0.533]]


def test_mh_2d_seed():
    target = Distributions.MultivariateNormal([0., 0.]).pdf
    mh_input = MhInput(pdf_target=target, seed=[0., 0.],
                       chains_number=1, random_state=123)
    x = MetropolisHastings(mh_input=mh_input, samples_number=10)
    assert [round(float(x.samples[-1][0]), 3), round(float(x.samples[-1][1]), 3)] == [-0.406, -1.217]


def test_mh_1d_symmetric_proposal_pdf():
    target = Distributions.Normal().pdf
    proposal = Distributions.Normal()
    mh_input = MhInput(dimension=1, pdf_target=target, proposal=proposal, proposal_is_symmetric=True,
                       chains_number=1, random_state=123)
    x = MetropolisHastings(mh_input=mh_input, samples_number=10)
    assert round(float(x.samples[-1]), 3) == -1.291


def test_mh_1d_asymmetric_proposal_pdf():
    target = Distributions.Normal().pdf
    proposal = Distributions.Normal()
    mh_input = MhInput(dimension=1, pdf_target=target, proposal=proposal, proposal_is_symmetric=False,
                       chains_number=1, random_state=123)
    x = MetropolisHastings(mh_input=mh_input, samples_number=10)
    assert round(float(x.samples[-1]), 3) == -1.291


def test_mmh_1d_burnjump():
    target = Distributions.Normal().pdf
    mmh_input = MmhInput(dimension=1, pdf_target=target, burn_length=10,
                         jump=2, chains_number=1, random_state=123)
    x = ModifiedMetropolisHastings(mmh_input=mmh_input, samples_number=10)
    assert round(float(x.samples[-1]), 3) == 0.497


def test_mmh_2d_list_target_pdf():
    target = [Distributions.Normal().pdf, Distributions.Normal().pdf]
    mmh_input = MmhInput(dimension=2, pdf_target=target, chains_number=1, random_state=123)
    x = ModifiedMetropolisHastings(mmh_input=mmh_input, samples_number=10)
    assert [round(float(x.samples[-1][0]), 3), round(float(x.samples[-1][1]), 3)] == [-0.810, 0.173]


def test_mmh_2d_list_target_logpdf():
    target = [Distributions.Normal().log_pdf, Distributions.Normal().log_pdf]
    mmh_input = MmhInput(dimension=2, log_pdf_target=target, chains_number=1, random_state=123)
    x = ModifiedMetropolisHastings(mmh_input=mmh_input, samples_number=10)
    assert [round(float(x.samples[-1][0]), 3), round(float(x.samples[-1][1]), 3)] == [-0.810, 0.173]


def test_mmh_2d_joint_proposal():
    target = Distributions.MultivariateNormal([0., 0.]).pdf
    proposal = Distributions.JointIndependent(marginals=[Distributions.Normal(scale=0.2),
                                                         Distributions.Normal(scale=0.2)])
    mmh_input = MmhInput(dimension=2, pdf_target=target, chains_number=1, proposal=proposal, random_state=123)
    x = ModifiedMetropolisHastings(mmh_input=mmh_input, samples_number=10)
    assert [round(float(x.samples[-1][0]), 3), round(float(x.samples[-1][1]), 3)] == [-0.783, -0.195]


def test_mmh_2d_list_proposal():
    target = Distributions.MultivariateNormal([0., 0.]).pdf
    proposal = [Distributions.Normal(scale=0.2), Distributions.Normal(scale=0.2)]
    mmh_input = MmhInput(dimension=2, pdf_target=target, chains_number=1, proposal=proposal, random_state=123)
    x = ModifiedMetropolisHastings(mmh_input=mmh_input, samples_number=10)
    assert [round(float(x.samples[-1][0]), 3), round(float(x.samples[-1][1]), 3)] == [-0.783, -0.195]


def test_mmh_2d_single1d_proposal():
    target = Distributions.MultivariateNormal([0., 0.]).pdf
    proposal = Distributions.Normal(scale=0.2)
    mmh_input = MmhInput(dimension=2, pdf_target=target, chains_number=1, proposal=proposal, random_state=123)
    x = ModifiedMetropolisHastings(mmh_input=mmh_input, samples_number=10)
    assert [round(float(x.samples[-1][0]), 3), round(float(x.samples[-1][1]), 3)] == [-0.783, -0.195]


def test_mmh_2d_list_proposal_log_target():
    target = [Distributions.Normal().log_pdf, Distributions.Normal().log_pdf]
    proposal = [Distributions.Normal(scale=0.2), Distributions.Normal(scale=0.2)]
    mmh_input = MmhInput(dimension=2, log_pdf_target=target, chains_number=1, proposal=proposal, random_state=123)
    x = ModifiedMetropolisHastings(mmh_input = mmh_input, samples_number=10)
    assert [round(float(x.samples[-1][0]), 3), round(float(x.samples[-1][1]), 3)] == [-0.783, -0.195]


def test_dram_1d_burnjump():
    target = Distributions.Normal().pdf
    dram_input = DramInput(dimension=1, pdf_target=target, burn_length=10, jump=2, chains_number=1, random_state=123)
    x = DRAM(dram_input=dram_input, samples_number=10)
    assert round(float(x.samples[-1]), 3) == 0.935


def test_dream_1d_burnjump():
    target = Distributions.Normal().pdf
    dreamInput = DreamInput(pdf_target=target, burn_length=10, jump=2, dimension=1, chains_number=10, random_state=123)
    x = DREAM(dream_input=dreamInput, samples_number=20)
    assert round(float(x.samples[-1]), 3) == 0.0


def test_dream_1d_checkchains():
    target = Distributions.Normal().pdf
    dreamInput = DreamInput(pdf_target=target, burn_length=0, jump=2, save_log_pdf=True,
                            dimension=1, check_chains=(1000, 1), chains_number=20, random_state=123)
    x = DREAM(dream_input=dreamInput, samples_number=2000)
    assert (round(float(x.samples[-1]), 3) == 0.593)


def test_dream_1d_adaptchains():
    target = Distributions.Normal().pdf
    dreamInput = DreamInput(pdf_target=target, burn_length=1000, jump=2, save_log_pdf=True,
                            dimension=1, crossover_adaptation=(1000, 1), chains_number=20, random_state=123)
    x = DREAM(dream_input=dreamInput, samples_number=2000)
    assert (round(float(x.samples[-1]), 3) == -0.446)


def test_stretch_1d_burnjump():
    target = Distributions.Normal().pdf
    stretch_input = StretchInput(pdf_target=target, burn_length=10, jump=2,
                                 dimension=1, chains_number=2, random_state=123)
    x = Stretch(stretch_input=stretch_input, samples_number=10)
    assert round(float(x.samples[-1]), 3) == -0.961


def test_unconcat_chains_mcmc():
    target = Distributions.Normal().pdf
    mmh_input = MmhInput(dimension=1, pdf_target=target, burn_length=10, jump=2,
                         chains_number=2, save_log_pdf=True, random_state=123)
    x = ModifiedMetropolisHastings(mmh_input=mmh_input)
    x.run(samples_number=5)
    x.run(samples_number=5)
    assert (round(float(x.samples[-1]), 3) == -0.744)
