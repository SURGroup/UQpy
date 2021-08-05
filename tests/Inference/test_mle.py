import numpy as np
from UQpy import MinimizeOptimizer
from UQpy.distributions.collection import Normal, JointCopula, JointIndependent
from UQpy.distributions.copulas.Gumbel import Gumbel
from UQpy.inference.inference_models.DistributionModel import DistributionModel
from UQpy.inference.inference_models.ComputationalModel import ComputationalModel
from UQpy.sampling.ImportanceSampling import ImportanceSampling
from UQpy.inference import MLE
from UQpy.RunModel import RunModel


def test_simple_probability_model():
    np.random.seed(1)
    mu, sigma = 0, 0.1  # true mean and standard deviation
    data_1 = np.random.normal(mu, sigma, 1000).reshape((-1, 1))
    # set parameters to be learnt as None
    dist = Normal(loc=None, scale=None)
    candidate_model = DistributionModel(distributions=dist, nparams=2)

    ml_estimator = MLE(inference_model=candidate_model, data=data_1, verbose=True, nopt=3, random_state=1)

    assert ml_estimator.mle[0] == 0.003881247615960185
    assert ml_estimator.mle[1] == 0.09810041339322118


# def test_complex_probability_model():
#     # dist_true exhibits dependence between the two dimensions, defined using a gumbel copula
#     dist_true = JointCopula(marginals=[Normal(), Normal()], copula=Gumbel(theta=2.))
#
#     # generate data using importance sampling: sample from a bivariate gaussian without copula, then weight samples
#     u = ImportanceSampling(proposal=JointIndependent(marginals=[Normal(), Normal()]),
#            log_pdf_target=dist_true.log_pdf, samples_number=500)
#     print(u.samples.shape)
#     print(u.weights.shape)
#     # Resample to obtain 5,000 data points
#     u.resample(samples_number=5000)
#     data_2 = u.unweighted_samples
#     print('Shape of data: {}'.format(data_2.shape))
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     ax.scatter(data_2[:, 0], data_2[:, 1], alpha=0.2)
#     ax.set_title('Data points from true bivariate normal with gumbel dependency structure')
#     plt.show()
#
#     d_guess = JointCopula(marginals=[Normal(loc=None, scale=None), Normal(loc=None, scale=None)],
#                           copula=Gumbel(theta=None))
#     print(d_guess.get_parameters())
#     candidate_model = DistributionModel(nparams=5, distributions=d_guess)
#     print(candidate_model.list_params)
#
#     ml_estimator = MLE(inference_model=candidate_model, data=data_2, verbose=True,
#                                 bounds=[[-5, 5], [0, 10], [-5, 5], [0, 10], [1.1, 4]], method='SLSQP')
#
#     ml_estimator.run(x0=[1., 1., 1., 1., 4.])
#
#     print('ML estimates of the mean={0:.3f} and std. dev={1:.3f} of 1st marginal (true: 0.0, 1.0)'.
#           format(ml_estimator.mle[0], ml_estimator.mle[1]))
#     print('ML estimates of the mean={0:.3f} and std. dev={1:.3f} of 2nd marginal (true: 0.0, 1.0)'.
#           format(ml_estimator.mle[2], ml_estimator.mle[3]))
#     print('ML estimates of the copula parameter={0:.3f} (true: 2.0)'.format(ml_estimator.mle[4]))
#
#     assert ml_estimator.mle[0] == 0.0
#     assert ml_estimator.mle[1] == 0.0

def test_regression_model():

    param_true = np.array([1.0, 2.0]).reshape((1, -1))

    h_func = RunModel(model_script='pfn_models.py', model_object_name='model_quadratic', vec=False,
                      var_names=['theta_0', 'theta_1'])
    h_func.run(samples=param_true)

    # Add noise
    error_covariance = 1.
    data_clean = np.array(h_func.qoi_list[0])
    noise = Normal(loc=0., scale=np.sqrt(error_covariance)).rvs(nsamples=50,random_state=1).reshape((50,))
    data_3 = data_clean + noise

    candidate_model = ComputationalModel(parameters_number=2, runmodel_object=h_func, error_covariance=error_covariance)

    optimizer=MinimizeOptimizer(method='nelder-mead')
    ml_estimator = MLE(inference_model=candidate_model, data=data_3, nopt=1, random_state=1, optimizer=optimizer)

    assert ml_estimator.mle[0] == 0.8822312624243898
    assert ml_estimator.mle[1] == 2.0164291038828983