from UQpy.SampleMethods.MCMC import *
import numpy as np
import UQpy.Distributions as Distributions


# Tests for parent MCMC and MH algorithm
def test_MH_1D_target_pdf():
    target = Distributions.Normal().pdf
    x = MH(dimension=1, pdf_target=target, nsamples=10, nchains=1, random_state=123)
    assert x.samples[-1] == [-1.2907929061320982]


def test_MH_1D_acceptancerate():
    target = Distributions.Normal().pdf
    x = MH(dimension=1, pdf_target=target, nsamples=100, nchains=1, random_state=123)
    assert x.acceptance_rate == [0.7070707070707071]


def test_MH_1D_savelogpdf():
    target = Distributions.Normal().pdf
    x = MH(dimension=1, pdf_target=target, nsamples=10, nchains=1, save_log_pdf=True, random_state=123)
    assert x.log_pdf_values[-1] == -1.7520116964651467


def test_MH_1D_target_logpdf():
    target = Distributions.Normal().log_pdf
    x = MH(dimension=1, log_pdf_target=target, nsamples=10, nchains=1, random_state=123)
    assert x.samples[-1] == [-1.2907929061320982]


def test_MH_2D():
    target = Distributions.MVNormal([0., 0.]).pdf
    x = MH(dimension=2, pdf_target=target, nsamples=10, nchains=1, random_state=123)
    assert (x.samples[-1] == [-0.4058509370712553, -1.2170691596327545]).all()


def test_MH_2D_burnjump():
    target = Distributions.MVNormal([0., 0.]).pdf
    x = MH(dimension=2, pdf_target=target, nburn=10, jump=2, nsamples=10, nchains=1, random_state=123)
    # y = MH(dimension=2, pdf_target=target, nsamples=31, nchains=1, random_state=123)
    assert (x.niterations == 30)


def test_MH_2D_nsamplecheck():
    target = Distributions.MVNormal([0., 0.]).pdf
    x = MH(dimension=2, pdf_target=target, nsamples=60, nchains=2, random_state=123)
    assert x.nsamples_per_chain + x.nsamples == 90


def test_MH_2D_2chainz():
    target = Distributions.MVNormal([0., 0.]).pdf
    x = MH(dimension=2, pdf_target=target, nsamples=60, nchains=2, random_state=123)
    assert (x.samples[-1] == [-0.06372740218182948, -0.5326418411320808]).all()


def test_MH_2D_2chainz_nonconcatenated():
    target = Distributions.MVNormal([0., 0.]).pdf
    x = MH(dimension=2, pdf_target=target, nsamples=60, nchains=2, concat_chains=False, random_state=123)
    assert (x.samples[-1] == [[1.7673280063103385, 1.4645434704726876], [-0.06372740218182948, -0.5326418411320808]]).all()


def test_MH_2D_seed():
    target = Distributions.MVNormal([0., 0.]).pdf
    x = MH(pdf_target=target, nsamples=10, nchains=1, seed=[0., 0.], random_state=123)
    assert (x.samples[-1] == [-0.4058509370712553, -1.2170691596327545]).all()


def test_MH_1D_symmetric_proposal_pdf():
    target = Distributions.Normal().pdf
    proposal = Distributions.Normal()
    x = MH(dimension=1, pdf_target=target, nsamples=10, nchains=1, proposal=proposal, proposal_is_symmetric=True,
           random_state=123)
    assert x.samples[-1] == [-1.2907929061320982]


