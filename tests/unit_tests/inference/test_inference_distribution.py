import numpy as np
from UQpy.inference import *
from UQpy.distributions import *
from UQpy.inference.inference_models.DistributionModel import DistributionModel

# data used throughout
from UQpy.inference.information_criteria import BIC, AICc
from UQpy.sampling.mcmc import MetropolisHastings

data = [0., 1., -1.5, -0.2]
# first candidate model, 1-dimensional
prior = Lognormal(s=1., loc=0., scale=1.)
dist = Normal(loc=0., scale=None)
candidate_model = DistributionModel(n_parameters=1, distributions=dist, prior=prior)
candidate_model_no_prior = DistributionModel(n_parameters=1, distributions=dist)
# second candidate model, 2-dimensional
prior2 = JointIndependent([Uniform(loc=0., scale=0.5), Lognormal(s=1., loc=0., scale=1.)])
dist2 = Uniform(loc=None, scale=None)
candidate_model2 = DistributionModel(n_parameters=2, distributions=dist2, prior=prior2)


def test_mle():
    ml_estimator = MLE(inference_model=candidate_model, data=data, n_optimizations=3)
    assert round(ml_estimator.mle[0], 3) == 0.907


def test_info_model_selection_bic():
    mle1 = MLE(inference_model=candidate_model, data=data)
    mle2 = MLE(inference_model=candidate_model2, data=data)
    selector = InformationModelSelection(parameter_estimators=[mle1, mle2], criterion=BIC(), n_optimizations=[5]*2)
    assert round(selector.probabilities[0], 3) == 0.284


def test_info_model_selection_aic():
    mle1 = MLE(inference_model=candidate_model, data=data)
    mle2 = MLE(inference_model=candidate_model2, data=data)
    selector = InformationModelSelection(parameter_estimators=[mle1, mle2], criterion=AIC(), n_optimizations=[5]*2)
    selector.sort_models()
    assert round(selector.probabilities[0], 3) == 0.650


def test_info_model_selection_aicc():
    mle1 = MLE(inference_model=candidate_model, data=data)
    mle2 = MLE(inference_model=candidate_model2, data=data)
    selector = InformationModelSelection(parameter_estimators=[mle1, mle2], criterion=AICc(), n_optimizations=[5]*2)
    assert round(selector.probabilities[0], 3) == 0.988


def test_bayes_mcmc():
    mh1 = MetropolisHastings(args_target=(data, ),
                             log_pdf_target=candidate_model_no_prior.evaluate_log_posterior,
                             jump=2, burn_length=5, seed=[1., ], random_state=123)
    bayes_estimator = BayesParameterEstimation(sampling_class=mh1, inference_model=candidate_model_no_prior,
                                               data=data, nsamples=50)
    assert np.round(np.mean(bayes_estimator.sampler.samples), 3) == 1.275


def test_bayes_is():
    is1 = ImportanceSampling(args_target=(data, ),
                             log_pdf_target=candidate_model.evaluate_log_posterior,
                             proposal=candidate_model.prior,
                             random_state=123)
    bayes_estimator = BayesParameterEstimation(sampling_class=is1, inference_model=candidate_model,
                                               data=data, nsamples=100)
    assert np.round(np.mean(bayes_estimator.sampler.samples), 3) == 1.873


def test_bayes_selection():
    mh1 = MetropolisHastings(args_target=(data, ),
                             log_pdf_target=candidate_model.evaluate_log_posterior,
                             random_state=123, n_chains=2, dimension=1)
    mh2 = MetropolisHastings(args_target=(data, ),
                             log_pdf_target=candidate_model2.evaluate_log_posterior,
                             random_state=123, n_chains=2, dimension=1)
    parameter_estimator = BayesParameterEstimation(inference_model=candidate_model, data=data, sampling_class=mh1)
    parameter_estimator1 = BayesParameterEstimation(inference_model=candidate_model2, data=data, sampling_class=mh2)
    selection = BayesModelSelection(parameter_estimators=[parameter_estimator, parameter_estimator1], nsamples=[50, 50])
    assert round(selection.probabilities[0], 3) == 1.000


def test_bayes_selection2():
    mh1 = MetropolisHastings(args_target=(data, ),
                             log_pdf_target=candidate_model.evaluate_log_posterior,
                             random_state=123, n_chains=2, dimension=1)
    mh2 = MetropolisHastings(args_target=(data, ),
                             log_pdf_target=candidate_model2.evaluate_log_posterior,
                             random_state=123, n_chains=2, dimension=1)
    parameter_estimator = BayesParameterEstimation(inference_model=candidate_model, data=data, sampling_class=mh1)
    parameter_estimator1 = BayesParameterEstimation(inference_model=candidate_model2, data=data, sampling_class=mh2)
    selection = BayesModelSelection(parameter_estimators=[parameter_estimator, parameter_estimator1], nsamples=[50, 50])
    selection.sort_models()
    assert round(selection.probabilities[0], 3) == 1.000
