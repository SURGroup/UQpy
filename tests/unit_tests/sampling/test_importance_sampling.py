import numpy as np
from UQpy.sampling import ImportanceSampling
from UQpy.distributions import JointIndependent, Uniform


def log_rosenbrock(x, param):
    return -(100 * (x[:, 1] - x[:, 0] ** 2) ** 2 + (1 - x[:, 0]) ** 2) / param


def rosenbrock(x):
    return np.exp(-(100 * (x[:, 1] - x[:, 0] ** 2) ** 2 + (1 - x[:, 0]) ** 2) / 20)


proposal = JointIndependent([Uniform(loc=-8, scale=16), Uniform(loc=-10, scale=60)])
proposal2 = JointIndependent([Uniform(loc=-8, scale=16), Uniform(loc=-10, scale=60)])
del proposal2.log_pdf


def test_pdf_target():
    w = ImportanceSampling(pdf_target=rosenbrock,
                           proposal=proposal,
                           random_state=123,
                           samples_number=2000)
    assert (w.weights.shape == (2000, ) and np.all(np.round(w.samples[-1], 3) == [-6.434, 27.373]))


def test_log_pdf_target():
    w = ImportanceSampling(log_pdf_target=log_rosenbrock,
                           args_target=(20, ),
                           proposal=proposal,
                           random_state=123,
                           samples_number=2000)
    assert (w.weights.shape == (2000, ) and np.all(np.round(w.samples[-1], 3) == [-6.434, 27.373]))


def test_resampling():
    w = ImportanceSampling(log_pdf_target=log_rosenbrock, args_target=(20,),
                           proposal=proposal, random_state=123,
                           samples_number=2000)
    w.resample(samples_number=1000)
    result = w.unweighted_samples[-1]
    assert np.all(np.round(result, 3) == [-4.912, 23.106])


def test_resampling2():
    w = ImportanceSampling(log_pdf_target=log_rosenbrock, args_target=(20,),
                           proposal=proposal, random_state=123, samples_number=2000)
    w.resample()
    assert w.unweighted_samples.shape == (2000, 2)


def test_rerun():
    w = ImportanceSampling(log_pdf_target=log_rosenbrock, args_target=(20,),
                           proposal=proposal, random_state=123)
    w.run(samples_number=1000)
    w.resample(samples_number=1000)
    w.run(samples_number=1000)
    assert w.samples.shape == (2000, 2)


def test_proposal():
    w = ImportanceSampling(log_pdf_target=log_rosenbrock, args_target=(20,),
                           proposal=proposal2, random_state=123, samples_number=2000)
    assert (w.weights.shape == (2000,) and np.all(np.round(w.samples[-1], 3) == [-6.434, 27.373]))

