import numpy as np
from UQpy.inference import *
from UQpy.distributions import *
from UQpy.inference.inference_models.DistributionModel import DistributionModel

# data used throughout
from UQpy.sampling.mcmc import MetropolisHastings

data = [0., 1., -1.5, -0.2]
# first candidate model, 1-dimensional
prior = Lognormal(s=1., loc=0., scale=1.)
dist = Normal(loc=0., scale=None)
candidate_model = DistributionModel(parameters_number=1, distributions=dist, prior=prior)
candidate_model_no_prior = DistributionModel(parameters_number=1, distributions=dist)
# second candidate model, 2-dimensional
prior2 = JointIndependent([Uniform(loc=0., scale=0.5), Lognormal(s=1., loc=0., scale=1.)])
dist2 = Uniform(loc=None, scale=None)
candidate_model2 = DistributionModel(parameters_number=2, distributions=dist2, prior=prior2)


def test_mle():
    ml_estimator = MLE(inference_model=candidate_model, data=data, optimizations_number=3)
    assert round(ml_estimator.mle[0], 3) == 0.907


def test_info_model_selection_bic():
    selector = InformationModelSelection(
        candidate_models=[candidate_model, candidate_model2], data=data,
        criterion=InformationTheoreticCriterion.BIC, optimizations_number=5)
    assert round(selector.probabilities[0], 3) == 0.284


def test_info_model_selection_aic():
    selector = InformationModelSelection(
        candidate_models=[candidate_model, candidate_model2], data=data,
        criterion=InformationTheoreticCriterion.AIC, optimizations_number=5)
    selector.sort_models()
    assert round(selector.probabilities[0], 3) == 0.650


def test_info_model_selection_aicc():
    selector = InformationModelSelection(
        candidate_models=[candidate_model, candidate_model2], data=data,
        criterion=InformationTheoreticCriterion.AICc, optimizations_number=5)
    assert round(selector.probabilities[0], 3) == 0.988


def test_bayes_mcmc():
    mh1 = MetropolisHastings.create_for_inference(inference_model=candidate_model_no_prior, data=data,
                                                  jump=2, burn_length=5, seed=[1., ], random_state=123)
    bayes_estimator = BayesParameterEstimation(sampling_class=mh1, inference_model=candidate_model_no_prior,
                                               data=data, samples_number=50)
    assert np.round(np.mean(bayes_estimator.sampler.samples), 3) == 1.275


def test_bayes_is():
    is1 = ImportanceSampling.create_for_inference(candidate_model, data, random_state=123)
    bayes_estimator = BayesParameterEstimation(sampling_class=is1, inference_model=candidate_model,
                                               data=data, samples_number=100)
    assert np.round(np.mean(bayes_estimator.sampler.samples), 3) == 1.873


def test_bayes_selection():
    mh1 = MetropolisHastings.create_for_inference(inference_model=candidate_model, data=data,
                                                  random_state=123, chains_number=2)
    mh2 = MetropolisHastings.create_for_inference(inference_model=candidate_model2, data=data,
                                                  random_state=123, chains_number=2)
    selection = BayesModelSelection(data=data, candidate_models=[candidate_model, candidate_model2],
                                    samples_number=[50, 50], sampling_class=[mh1, mh2])
    assert round(selection.probabilities[0], 3) == 1.000


def test_bayes_selection2():
    mh1 = MetropolisHastings.create_for_inference(inference_model=candidate_model, data=data,
                                                  random_state=123, chains_number=2)
    mh2 = MetropolisHastings.create_for_inference(inference_model=candidate_model2, data=data,
                                                  random_state=123, chains_number=2)
    selection = BayesModelSelection(data=data, candidate_models=[candidate_model, candidate_model2],
                                    samples_per_chain_number=[25, 25], sampling_class=[mh1, mh2])
    selection.sort_models()
    assert round(selection.probabilities[0], 3) == 1.000
