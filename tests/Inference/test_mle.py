import numpy as np

from UQpy.inference.inference_models.optimization.MinizeOptimizer import MinimizeOptimizer
from UQpy.distributions.collection import Normal, JointCopula, JointIndependent
from UQpy.distributions.copulas.Gumbel import Gumbel
from UQpy.inference.inference_models.DistributionModel import DistributionModel
from UQpy.inference.inference_models.ComputationalModel import ComputationalModel
from UQpy.sampling.ImportanceSampling import ImportanceSampling
from UQpy.inference import MLE
from UQpy.RunModel import RunModel
import shutil


def test_simple_probability_model():
    np.random.seed(1)
    mu, sigma = 0, 0.1  # true mean and standard deviation
    data_1 = np.random.normal(mu, sigma, 1000).reshape((-1, 1))
    # set parameters to be learnt as None
    dist = Normal(loc=None, scale=None)
    candidate_model = DistributionModel(distributions=dist, parameters_number=2)

    ml_estimator = MLE(inference_model=candidate_model, data=data_1, optimizations_number=3, random_state=1)

    assert ml_estimator.mle[0] == 0.003881247615960185
    assert ml_estimator.mle[1] == 0.09810041339322118


def test_regression_model():

    param_true = np.array([1.0, 2.0]).reshape((1, -1))

    h_func = RunModel(model_script='pfn_models.py', model_object_name='model_quadratic', vec=False,
                      var_names=['theta_0', 'theta_1'])
    h_func.run(samples=param_true)

    # Add noise
    error_covariance = 1.
    data_clean = np.array(h_func.qoi_list[0])
    noise = Normal(loc=0., scale=np.sqrt(error_covariance)).rvs(nsamples=4,random_state=1).reshape((4,))
    data_3 = data_clean + noise

    candidate_model = ComputationalModel(parameters_number=2, runmodel_object=h_func, error_covariance=error_covariance)

    optimizer = MinimizeOptimizer(method='nelder-mead')
    ml_estimator = MLE(inference_model=candidate_model, data=data_3, optimizations_number=1, random_state=1,
                       optimizer=optimizer)

    assert ml_estimator.mle[0] == 0.8689097631871134
    assert ml_estimator.mle[1] == 2.0030767805841143

    shutil.rmtree(h_func.model_dir)
