import numpy as np
from UQpy.SampleMethods import IS
from UQpy.Distributions import JointInd, Uniform


def log_rosenbrock(x, param):
    return -(100 * (x[:, 1] - x[:, 0] ** 2) ** 2 + (1 - x[:, 0]) ** 2) / param


def rosenbrock(x):
    return np.exp(-(100 * (x[:, 1] - x[:, 0] ** 2) ** 2 + (1 - x[:, 0]) ** 2) / 20)


proposal = JointInd([Uniform(loc=-8, scale=16), Uniform(loc=-10, scale=60)])
proposal2 = JointInd([Uniform(loc=-8, scale=16), Uniform(loc=-10, scale=60)])
del proposal2.log_pdf


def test_pdf_target():
    w = IS(pdf_target=rosenbrock, nsamples=2000, proposal=proposal, random_state=123)
    assert (w.weights.shape == (2000, ) and np.all(np.round(w.samples[-1], 3) == [-6.434, 27.373]))


def test_log_pdf_target():
    w = IS(log_pdf_target=log_rosenbrock, args_target=(20, ), nsamples=2000, proposal=proposal, random_state=123)
    assert (w.weights.shape == (2000, ) and np.all(np.round(w.samples[-1], 3) == [-6.434, 27.373]))


def test_resampling():
    w = IS(log_pdf_target=log_rosenbrock, args_target=(20, ), nsamples=2000, proposal=proposal, random_state=123)
    w.resample(nsamples=1000)
    assert np.all(np.round(w.unweighted_samples[-1], 3) == [-4.912, 23.106])


def test_resampling2():
    w = IS(log_pdf_target=log_rosenbrock, args_target=(20, ), nsamples=2000, proposal=proposal, random_state=123)
    w.resample()
    assert w.unweighted_samples.shape == (2000, 2)


def test_rerun():
    w = IS(log_pdf_target=log_rosenbrock, args_target=(20,), proposal=proposal, random_state=123)
    w.run(nsamples=1000)
    w.resample(nsamples=1000)
    w.run(nsamples=1000)
    assert w.samples.shape == (2000, 2)


def test_proposal():
    w = IS(log_pdf_target=log_rosenbrock, args_target=(20,), proposal=proposal2, random_state=123, nsamples=2000)
    assert (w.weights.shape == (2000,) and np.all(np.round(w.samples[-1], 3) == [-6.434, 27.373]))

