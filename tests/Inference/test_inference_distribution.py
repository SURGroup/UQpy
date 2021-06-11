import numpy as np
from UQpy.Inference import *
from UQpy.Distributions import Normal, Uniform, Lognormal, JointInd
from UQpy.SampleMethods import MH, IS

# data used throughout
data = [0., 1., -1.5, -0.2]
# first candidate model, 1-dimensional
prior = Lognormal(s=1., loc=0., scale=1.)
dist = Normal(loc=0., scale=None)
candidate_model = InferenceModel(nparams=1, dist_object=dist, prior=prior)
candidate_model_no_prior = InferenceModel(nparams=1, dist_object=dist)
# second candidate model, 2-dimensional
prior2 = JointInd([Uniform(loc=0., scale=0.5), Lognormal(s=1., loc=0., scale=1.)])
dist2 = Uniform(loc=None, scale=None)
candidate_model2 = InferenceModel(nparams=2, dist_object=dist2, prior=prior2)


def test_mle():
    ml_estimator = MLEstimation(inference_model=candidate_model, data=data, nopt=3)
    assert round(ml_estimator.mle[0], 3) == 0.907


def test_infomodelselection_bic():
    selector = InfoModelSelection(
        candidate_models=[candidate_model, candidate_model2], data=data, criterion='BIC', nopt=5)
    assert round(selector.probabilities[0], 3) == 0.284


def test_infomodelselection_aic():
    selector = InfoModelSelection(
        candidate_models=[candidate_model, candidate_model2], data=data, criterion='AIC', nopt=5)
    selector.sort_models()
    assert round(selector.probabilities[0], 3) == 0.650


def test_infomodelselection_aicc():
    selector = InfoModelSelection(
        candidate_models=[candidate_model, candidate_model2], data=data, criterion='AICc', nopt=5)
    assert round(selector.probabilities[0], 3) == 0.988


def test_bayes_mcmc():
    bayes_estimator = BayesParameterEstimation(
        data=data, inference_model=candidate_model_no_prior, sampling_class=MH, nsamples=50, jump=2, nburn=5,
        random_state=123, seed=[1., ])
    assert np.round(np.mean(bayes_estimator.sampler.samples), 3) == 1.275


def test_bayes_is():
    bayes_estimator = BayesParameterEstimation(
        data=data, inference_model=candidate_model, sampling_class=IS, nsamples=100, random_state=123)
    assert np.round(np.mean(bayes_estimator.sampler.samples), 3) == 1.873


def test_bayes_selection():
    selection = BayesModelSelection(
        data=data, candidate_models=[candidate_model, candidate_model2],
        nsamples=[50, 50], sampling_class=[MH, ] * 2, nchains=[2, 2], random_state=123)
    assert round(selection.probabilities[0], 3) == 1.000


def test_bayes_selection2():
    selection = BayesModelSelection(
        data=data, candidate_models=[candidate_model, candidate_model2],
        nsamples_per_chain=[25, 25], sampling_class=[MH, ] * 2, nchains=[2, 2], random_state=123)
    selection.sort_models()
    assert round(selection.probabilities[0], 3) == 1.000
