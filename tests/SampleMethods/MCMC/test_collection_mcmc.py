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
    mh_input = MhInput()
    mh_input.dimension = 1
    mh_input.pdf_target = target
    mh_input.chains_number = 1
    mh_input.random_state = 123
    x = MetropolisHastings(mh_input=mh_input, samples_number=10)
    assert round(float(x.samples[-1]), 3) == -1.291


def test_mh_1d_samplesperchain():
    target = Distributions.Normal().pdf
    mh_input = MhInput()
    mh_input.dimension = 1
    mh_input.pdf_target = target
    mh_input.chains_number = 2
    mh_input.random_state = 123
    x = MetropolisHastings(mh_input=mh_input, samples_number_per_chain=5)
    assert round(float(x.samples[-1]), 3) == 0.474


def test_mh_1d_acceptancerate():
    target = Distributions.Normal().pdf
    mh_input = MhInput()
    mh_input.dimension = 1
    mh_input.pdf_target = target
    mh_input.chains_number = 1
    mh_input.random_state = 123
    x = MetropolisHastings(mh_input=mh_input, samples_number=100)
    assert round(float(x.acceptance_rate[0]), 3) == 0.707


def test_mh_1d_savelogpdf():
    target = Distributions.Normal().pdf
    mh_input = MhInput()
    mh_input.dimension = 1
    mh_input.pdf_target = target
    mh_input.chains_number = 1
    mh_input.random_state = 123
    mh_input.save_log_pdf = True
    x = MetropolisHastings(mh_input=mh_input, samples_number=10)
    assert round(float(x.log_pdf_values[-1]), 3) == -1.752


def test_mh_1d_target_logpdf():
    target = Distributions.Normal().log_pdf
    mh_input = MhInput()
    mh_input.dimension = 1
    mh_input.log_pdf_target = target
    mh_input.chains_number = 1
    mh_input.random_state = 123
    x = MetropolisHastings(mh_input=mh_input, samples_number=10)
    assert round(float(x.samples[-1]), 3) == -1.291


def test_mh_2d():
    target = Distributions.MultivariateNormal([0., 0.]).pdf
    mh_input = MhInput()
    mh_input.dimension = 2
    mh_input.pdf_target = target
    mh_input.chains_number = 1
    mh_input.random_state = 123
    x = MetropolisHastings(mh_input=mh_input, samples_number=10)
    assert [round(float(x.samples[-1][0]), 3), round(float(x.samples[-1][1]), 3)] == [-0.406, -1.217]


def test_mh_2d_burnjump():
    target = Distributions.MultivariateNormal([0., 0.]).pdf
    mh_input = MhInput()
    mh_input.dimension = 2
    mh_input.pdf_target = target
    mh_input.burn_length = 10
    mh_input.jump = 2
    mh_input.chains_number = 1
    mh_input.random_state = 123
    x = MetropolisHastings(mh_input=mh_input, samples_number=10)
    assert x.iterations_number == 30


def test_mh_2d_nsamplecheck():
    target = Distributions.MultivariateNormal([0., 0.]).pdf
    mh_input = MhInput()
    mh_input.dimension = 2
    mh_input.pdf_target = target
    mh_input.chains_number = 2
    mh_input.random_state = 123
    x = MetropolisHastings(mh_input = mh_input, samples_number=60)
    assert x.nsamples_per_chain + x.samples_number == 90


def test_mh_2d_2chainz():
    target = Distributions.MultivariateNormal([0., 0.]).pdf
    mh_input = MhInput()
    mh_input.dimension = 2
    mh_input.pdf_target = target
    mh_input.chains_number = 2
    mh_input.random_state = 123
    x = MetropolisHastings(mh_input=mh_input, samples_number=60)
    assert [round(float(x.samples[-1][0]), 3), round(float(x.samples[-1][1]), 3)] == [-0.064, -0.533]


def test_mh_2d_2chainz_nonconcatenated():
    target = Distributions.MultivariateNormal([0., 0.]).pdf
    mh_input = MhInput()
    mh_input.dimension = 2
    mh_input.pdf_target = target
    mh_input.chains_number = 2
    mh_input.concatenate_chains = False
    mh_input.random_state = 123
    x = MetropolisHastings(mh_input=mh_input, samples_number=60)
    assert [[round(float(x.samples[-1][0][0]), 3), round(float(x.samples[-1][0][1]), 3)], [round(float(x.samples[-1][1][0]), 3), round(float(x.samples[-1][1][1]), 3)]] == [[1.767, 1.465], [-0.064, -0.533]]


def test_mh_2d_seed():
    target = Distributions.MultivariateNormal([0., 0.]).pdf
    mh_input = MhInput()
    mh_input.pdf_target = target
    mh_input.chains_number = 1
    mh_input.seed = [0., 0.]
    mh_input.random_state = 123
    x = MetropolisHastings(mh_input=mh_input, samples_number=10)
    assert [round(float(x.samples[-1][0]), 3), round(float(x.samples[-1][1]), 3)] == [-0.406, -1.217]


def test_mh_1d_symmetric_proposal_pdf():
    target = Distributions.Normal().pdf
    proposal = Distributions.Normal()
    mh_input = MhInput()
    mh_input.dimension = 1
    mh_input.pdf_target = target
    mh_input.chains_number = 1
    mh_input.proposal = proposal
    mh_input.proposal_is_symmetric = True
    mh_input.random_state = 123
    x = MetropolisHastings(mh_input=mh_input, samples_number=10)
    assert round(float(x.samples[-1]), 3) == -1.291


def test_mh_1d_asymmetric_proposal_pdf():
    target = Distributions.Normal().pdf
    proposal = Distributions.Normal()
    mh_input = MhInput()
    mh_input.dimension = 1
    mh_input.pdf_target = target
    mh_input.chains_number = 1
    mh_input.proposal = proposal
    mh_input.proposal_is_symmetric = False
    mh_input.random_state = 123
    x = MetropolisHastings(mh_input=mh_input, samples_number=10)
    assert round(float(x.samples[-1]), 3) == -1.291


def test_mmh_1d_burnjump():
    target = Distributions.Normal().pdf
    mmh_input = MmhInput()
    mmh_input.dimension = 1
    mmh_input.pdf_target = target
    mmh_input.burn_length = 10
    mmh_input.jump = 2
    mmh_input.chains_number = 1
    mmh_input.random_state = 123
    x = ModifiedMetropolisHastings(mmh_input=mmh_input, samples_number=10)
    assert round(float(x.samples[-1]), 3) == 0.497


def test_mmh_2d_list_target_pdf():
    target = [Distributions.Normal().pdf, Distributions.Normal().pdf]
    mmh_input = MmhInput()
    mmh_input.dimension = 2
    mmh_input.pdf_target = target
    mmh_input.chains_number = 1
    mmh_input.random_state = 123
    x = ModifiedMetropolisHastings(mmh_input=mmh_input, samples_number=10)
    assert [round(float(x.samples[-1][0]), 3), round(float(x.samples[-1][1]), 3)] == [-0.810, 0.173]


def test_mmh_2d_list_target_logpdf():
    target = [Distributions.Normal().log_pdf, Distributions.Normal().log_pdf]
    mmh_input = MmhInput()
    mmh_input.dimension = 2
    mmh_input.log_pdf_target = target
    mmh_input.chains_number = 1
    mmh_input.random_state = 123
    x = ModifiedMetropolisHastings(mmh_input=mmh_input, samples_number=10)
    assert [round(float(x.samples[-1][0]), 3), round(float(x.samples[-1][1]), 3)] == [-0.810, 0.173]


def test_mmh_2d_joint_proposal():
    target = Distributions.MultivariateNormal([0., 0.]).pdf
    proposal = Distributions.JointIndependent(marginals=[Distributions.Normal(scale=0.2),
                                                         Distributions.Normal(scale=0.2)])
    mmh_input = MmhInput()
    mmh_input.dimension = 2
    mmh_input.pdf_target = target
    mmh_input.chains_number = 1
    mmh_input.proposal = proposal
    mmh_input.random_state = 123
    x = ModifiedMetropolisHastings(mmh_input=mmh_input, samples_number=10)
    assert [round(float(x.samples[-1][0]), 3), round(float(x.samples[-1][1]), 3)] == [-0.783, -0.195]


def test_mmh_2d_list_proposal():
    target = Distributions.MultivariateNormal([0., 0.]).pdf
    proposal = [Distributions.Normal(scale=0.2), Distributions.Normal(scale=0.2)]
    mmh_input = MmhInput()
    mmh_input.dimension = 2
    mmh_input.pdf_target = target
    mmh_input.chains_number = 1
    mmh_input.proposal = proposal
    mmh_input.random_state = 123
    x = ModifiedMetropolisHastings(mmh_input=mmh_input, samples_number=10)
    assert [round(float(x.samples[-1][0]), 3), round(float(x.samples[-1][1]), 3)] == [-0.783, -0.195]


def test_mmh_2d_single1d_proposal():
    target = Distributions.MultivariateNormal([0., 0.]).pdf
    proposal = Distributions.Normal(scale=0.2)
    mmh_input = MmhInput()
    mmh_input.dimension = 2
    mmh_input.pdf_target = target
    mmh_input.chains_number = 1
    mmh_input.proposal = proposal
    mmh_input.random_state = 123
    x = ModifiedMetropolisHastings(mmh_input=mmh_input, samples_number=10)
    assert [round(float(x.samples[-1][0]), 3), round(float(x.samples[-1][1]), 3)] == [-0.783, -0.195]


def test_mmh_2d_list_proposal_log_target():
    target = [Distributions.Normal().log_pdf, Distributions.Normal().log_pdf]
    proposal = [Distributions.Normal(scale=0.2), Distributions.Normal(scale=0.2)]
    mmh_input = MmhInput()
    mmh_input.dimension = 2
    mmh_input.log_pdf_target = target
    mmh_input.chains_number = 1
    mmh_input.proposal = proposal
    mmh_input.random_state = 123
    x = ModifiedMetropolisHastings(mmh_input = mmh_input, samples_number=10)
    assert [round(float(x.samples[-1][0]), 3), round(float(x.samples[-1][1]), 3)] == [-0.783, -0.195]


def test_dram_1d_burnjump():
    target = Distributions.Normal().pdf
    dram_input = DramInput()
    dram_input.dimension = 1
    dram_input.pdf_target = target
    dram_input.burn_length = 10
    dram_input.jump = 2
    dram_input.chains_number = 1
    dram_input.random_state = 123
    x = DRAM(dram_input=dram_input, samples_number=10)
    assert round(float(x.samples[-1]), 3) == 0.935


def test_dream_1d_burnjump():
    target = Distributions.Normal().pdf
    dreamInput = DreamInput()
    dreamInput.pdf_target = target
    dreamInput.burn_length = 10
    dreamInput.jump = 2
    dreamInput.dimension = 1
    dreamInput.chains_number = 10
    dreamInput.random_state = 123
    x = DREAM(dream_input=dreamInput, samples_number=20)
    assert round(float(x.samples[-1]), 3) == 0.0


def test_dream_1d_checkchains():
    target = Distributions.Normal().pdf
    dreamInput = DreamInput()
    dreamInput.pdf_target = target
    dreamInput.burn_length = 0
    dreamInput.jump = 2
    dreamInput.save_log_pdf = True
    dreamInput.dimension = 1
    dreamInput.check_chains = (1000, 1)
    dreamInput.chains_number = 20
    dreamInput.random_state = 123
    x = DREAM(dream_input=dreamInput, samples_number=2000)
    assert (round(float(x.samples[-1]), 3) == 0.593)


def test_dream_1d_adaptchains():
    target = Distributions.Normal().pdf
    dreamInput = DreamInput()
    dreamInput.pdf_target = target
    dreamInput.burn_length = 1000
    dreamInput.jump = 2
    dreamInput.save_log_pdf = True
    dreamInput.dimension = 1
    dreamInput.crossover_adaptation = (1000, 1)
    dreamInput.chains_number = 20
    dreamInput.random_state = 123
    x = DREAM(dream_input=dreamInput, samples_number=2000)
    assert (round(float(x.samples[-1]), 3) == -0.446)


def test_stretch_1d_burnjump():
    target = Distributions.Normal().pdf
    stretch_input = StretchInput()
    stretch_input.pdf_target = target
    stretch_input.burn_length = 10
    stretch_input.jump = 2
    stretch_input.dimension = 1
    stretch_input.chains_number = 2
    stretch_input.random_state = 123
    x = Stretch(stretch_input=stretch_input, samples_number=10)
    assert round(float(x.samples[-1]), 3) == -0.961


def test_unconcat_chains_mcmc():
    target = Distributions.Normal().pdf
    mmh_input = MmhInput()
    mmh_input.dimension = 1
    mmh_input.pdf_target = target
    mmh_input.burn_length = 10
    mmh_input.jump = 2
    mmh_input.chains_number = 2
    mmh_input.save_log_pdf = True
    mmh_input.random_state = 123
    x = ModifiedMetropolisHastings(mmh_input=mmh_input)
    x.run(samples_number=5)
    x.run(samples_number=5)
    assert (round(float(x.samples[-1]), 3) == -0.744)
