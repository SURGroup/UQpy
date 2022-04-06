from UQpy.run_model import PythonModel, RunModel_New
from UQpy.inference import *
from UQpy.inference.inference_models.ComputationalModel import ComputationalModel
from UQpy.sampling.mcmc import MetropolisHastings
from UQpy.distributions.collection.Normal import Normal
from UQpy.distributions.collection.JointIndependent import JointIndependent
import shutil

import os

dir_path = os.path.dirname(os.path.realpath(__file__))
# print(dir_path)
# print(os.getcwd())
# os.chdir(dir_path)
# print(os.getcwd())


# os.chdir("~/test/unit_tests/inference")


def test_models():
    a = os.getcwd()
    if "inference" not in a:
        raise NotImplementedError(a)
    data_ex1 = np.loadtxt('data_ex1a.txt')

    model = PythonModel(model_script='pfn_linear.py', model_object_name='model_linear', var_names=['theta_0'])
    runmodel4 = RunModel_New(model=model)

    model1 = PythonModel(model_script='pfn_quadratic.py', model_object_name='model_quadratic', var_names=['theta_0', 'theta_1'])
    runmodel5 = RunModel_New(model=model1)

    model2 = PythonModel(model_script='pfn_cubic.py', model_object_name='model_cubic',
                         var_names=['theta_0', 'theta_1', 'theta_2'])
    runmodel6 = RunModel_New(model=model2)

    prior1 = Normal()
    prior2 = JointIndependent(marginals=[Normal(), Normal()])
    prior3 = JointIndependent(marginals=[Normal(), Normal(), Normal()])

    model_n_params = [1, 2, 3]
    model1 = ComputationalModel(n_parameters=1, runmodel_object=runmodel4, prior=prior1,
                                error_covariance=np.ones(50), name='model_linear')
    model2 = ComputationalModel(n_parameters=2, runmodel_object=runmodel5, prior=prior2,
                                error_covariance=np.ones(50), name='model_quadratic')
    model3 = ComputationalModel(n_parameters=3, runmodel_object=runmodel6, prior=prior3,
                                error_covariance=np.ones(50), name='model_cubic')

    proposals = [Normal(0, 10),
                 JointIndependent([Normal(0, 1), Normal(0, 1)]),
                 JointIndependent([Normal(0, 1), Normal(0, 2), Normal(0.025)])]

    # sampling =
    mh1 = MetropolisHastings(args_target=(data_ex1,),
                             log_pdf_target=model1.evaluate_log_posterior,
                             jump=1, burn_length=500,
                             proposal=proposals[0], random_state=0, seed=[0.])
    mh2 = MetropolisHastings(args_target=(data_ex1,),
                             log_pdf_target=model2.evaluate_log_posterior,
                             jump=1, burn_length=500,
                             proposal=proposals[1], random_state=0, seed=[0., 0.])
    mh3 = MetropolisHastings(args_target=(data_ex1,),
                             log_pdf_target=model3.evaluate_log_posterior,
                             jump=1, burn_length=500,
                             proposal=proposals[2], random_state=0, seed=[0., 0., 0.])

    e1 = BayesParameterEstimation(inference_model=model1, data=data_ex1, sampling_class=mh1)
    e2 = BayesParameterEstimation(inference_model=model2, data=data_ex1, sampling_class=mh2)
    e3 = BayesParameterEstimation(inference_model=model3, data=data_ex1, sampling_class=mh3)

    selection = BayesModelSelection(parameter_estimators=[e1, e2, e3],
                                    prior_probabilities=[1. / 3., 1. / 3., 1. / 3.],
                                    nsamples=[2000, 2000, 2000])

    selection.sort_models()
    assert selection.probabilities[0] == 1.0
    assert selection.probabilities[1] == 0.0
    assert selection.probabilities[2] == 0.0

    assert selection.candidate_models[0].name == 'model_quadratic'
    assert selection.candidate_models[1].name == 'model_cubic'
    assert selection.candidate_models[2].name == 'model_linear'

