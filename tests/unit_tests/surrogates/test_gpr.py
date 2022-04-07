import pytest
from beartype.roar import BeartypeCallHintPepParamException

from UQpy.utilities.MinimizeOptimizer import MinimizeOptimizer
from UQpy.utilities.FminCobyla import FminCobyla
from UQpy.surrogates.gaussian_process.GaussianProcessRegression import GaussianProcessRegression
# # from UQpy.utilities.strata.Rectangular import Rectangular
# # from UQpy.sampling.StratifiedSampling import StratifiedSampling
# # from UQpy.RunModel import RunModel
# # from UQpy.distributions.collection.Uniform import Uniform
import numpy as np
import shutil
from UQpy.surrogates.gaussian_process.constraints import NonNegative
from UQpy.surrogates.gaussian_process.regression_models import LinearRegression, ConstantRegression, QuadraticRegression
from UQpy.surrogates.gaussian_process.kernels import RBF, Matern


samples = np.linspace(0, 5, 20).reshape(-1, 1)
values = np.cos(samples)
optimizer = MinimizeOptimizer(method="L-BFGS-B", bounds=[[0.1, 5], [0.1, 3]])
gpr = GaussianProcessRegression(kernel=RBF(), hyperparameters=[2.852, 2.959], random_state=1)
gpr.fit(samples=samples, values=values)

optimizer1 = MinimizeOptimizer(method="L-BFGS-B")
gpr2 = GaussianProcessRegression(kernel=Matern(nu=0.5), hyperparameters=[10, 4], optimizer=optimizer1,
                                 bounds=[[0.01, 100], [0.1, 10]], optimizations_number=10, random_state=2)
gpr2.fit(samples=samples, values=values, optimizations_number=10)

points11 = np.array([[0], [1], [2]])
points12 = np.array([[-1], [0], [-1]])
hyperparameters1 = np.array([1.5, 10])

points21 = np.array([[0, 0], [1, 1], [2, 2]])
points22 = np.array([[-1, 0], [1, 2], [0, -2]])
hyperparameters2 = np.array([3, 1, 2])
rbf_kernel = RBF()
matern_kernel_1_2 = Matern(nu=0.5)
matern_kernel_3_2 = Matern(nu=1.5)
matern_kernel_5_2 = Matern(nu=2.5)
matern_kernel_inf = Matern(nu=np.inf)
matern_kernel_2_1 = Matern(nu=2)

constant_reg = ConstantRegression()
linear_reg = LinearRegression()
quadratic_reg = QuadraticRegression()

non_negative = NonNegative(constraint_points=np.array([[1], [2], [3]]), observed_error=0.1, z_value=2)
optimizer3 = FminCobyla()
gpr3 = GaussianProcessRegression(regression_model=constant_reg, kernel=RBF(), hyperparameters=[2.852, 2.959],
                                 random_state=1, optimizer=optimizer3, optimize_constraints=non_negative,
                                 bounds=[[0.1, 5], [0.1, 3]])
gpr3.fit(samples=samples, values=values+1.05)

gpr4 = GaussianProcessRegression(kernel=RBF(), hyperparameters=[2.852, 2.959, 0.001], random_state=1,
                                 optimizer=optimizer3, bounds=[[0.1, 5], [0.1, 3], [1e-6, 1e-1]], noise=True)
gpr4.fit(samples=samples, values=values)



def test_predict1():
    prediction = np.round(gpr.predict([[1], [np.pi/2], [np.pi]], True), 3)
    expected_prediction = np.array([[0.54, 0., -1.], [0.,  0.,  0.]])
    assert (expected_prediction == prediction).all()


def test_predict2():
    prediction = np.round(gpr2.predict([[1], [2*np.pi], [np.pi]], True), 3)
    expected_prediction = np.array([[0.537,  0.238, -0.998], [0.077,  0.39,  0.046]])
    assert (expected_prediction == prediction).all()


def test_predict3():
    prediction = np.round(gpr3.predict([[1], [2*np.pi], [np.pi]], True), 3)
    expected_prediction = np.array([[1.59, 2.047, 0.05], [0., 0.006, 0.]])
    assert (expected_prediction == prediction).all()


def test_predict4():
    prediction = np.round(gpr4.predict([[1], [2 * np.pi], [np.pi]], True), 3)
    expected_prediction = np.array([[0.54,  0.983, -1.], [0.,  0.04,  0.]])
    assert np.isclose(prediction, expected_prediction, atol=0.05).all()


def test_rbf():
    """
    Test RBF kernel for 1-d input model
    """
    cov = rbf_kernel.c(points11, points12, hyperparameters1)
    covariance_matrix = np.round(cov, 3)
    expected_covariance_matrix = np.array([[80.074, 100.,  80.074], [41.111,  80.074,  41.111],
                                           [13.534,  41.111,  13.534]])
    assert (expected_covariance_matrix == covariance_matrix).all()


def test_rbf2():
    """
    Test RBF kernel for 2-d input model
    """
    cov = rbf_kernel.c(points21, points22, hyperparameters2)
    covariance_matrix2 = np.round(cov, 3)
    expected_covariance_matrix2 = np.array([[3.784e+00, 5.120e-01, 5.410e-01], [1.943e+00, 2.426e+00, 4.200e-02],
                                            [3.280e-01, 3.784e+00, 1.000e-03]])
    assert (expected_covariance_matrix2 == covariance_matrix2).all()


def test_matern_inf():
    """
    Test Matern kernel (nu=inf) for 1-d input model, should concide with RBF kernel
    """
    cov = matern_kernel_inf.c(points11, points12, hyperparameters1)
    covariance_matrix = np.round(cov, 3)
    expected_covariance_matrix = np.array([[80.074, 100.,  80.074], [41.111,  80.074,  41.111],
                                           [13.534,  41.111,  13.534]])
    assert (expected_covariance_matrix == covariance_matrix).all()


def tets_matern_inf2():
    """
    Test Matern kernel (nu=inf) for 2-d input model
    """
    cov = matern_kernel_inf.c(points21, points22, hyperparameters2)
    covariance_matrix2 = np.round(cov, 3)
    expected_covariance_matrix2 = np.array([[3.784e+00, 5.120e-01, 5.410e-01], [1.943e+00, 2.426e+00, 4.200e-02],
                                            [3.280e-01, 3.784e+00, 1.000e-03]])
    assert (expected_covariance_matrix2 == covariance_matrix2).all()


def test_matern_1_2():
    """
    Test Matern kernel (nu=0.5) for 1-d input model, should concide with absolute exponential kernel
    """
    cov = matern_kernel_1_2.c(points11, points12, hyperparameters1)
    covariance_matrix = np.round(cov, 3)
    expected_covariance_matrix = np.array([[51.342, 100.,  51.342], [26.36,  51.342,  26.36],
                                           [13.534,  26.36,  13.534]])
    assert (expected_covariance_matrix == covariance_matrix).all()


def test_matern_1_2_2():
    """
    Test Matern kernel (nu=0.5) for 2-d input model
    """
    cov = matern_kernel_1_2.c(points21, points22, hyperparameters2)
    covariance_matrix2 = np.round(cov, 3)
    expected_covariance_matrix2 = np.array([[2.866, 0.527, 0.541], [1.203, 1.472, 0.196], [0.428, 2.866, 0.069]])
    assert (expected_covariance_matrix2 == covariance_matrix2).all()


def test_matern_3_2():
    """
    Test Matern kernel (nu=1.5) for 1-d input model
    """
    cov = matern_kernel_3_2.c(points11, points12, hyperparameters1)
    covariance_matrix = np.round(cov, 3)
    expected_covariance_matrix = np.array([[67.906, 100.,  67.906], [32.869,  67.906,  32.869],
                                           [13.973,  32.869,  13.973]])
    assert (expected_covariance_matrix == covariance_matrix).all()


def test_matern_5_2():
    """
    Test Matern kernel (nu=2.5) for 1-d input model
    """
    cov = matern_kernel_5_2.c(points11, points12, hyperparameters1)
    covariance_matrix = np.round(cov, 3)
    expected_covariance_matrix = np.array([[72.776, 100.,  72.776], [35.222,  72.776,  35.222],
                                           [13.866,  35.222,  13.866]])
    assert (expected_covariance_matrix == covariance_matrix).all()


def test_matern_2_1():
    """
    Test Matern kernel (nu=2) for 1-d input model
    """
    cov = matern_kernel_2_1.c(points11, points12, hyperparameters1)
    covariance_matrix = np.round(cov, 3)
    expected_covariance_matrix = np.array([[70.894, np.nan,  70.894], [34.25, 70.894, 34.25],
                                           [13.921, 34.25, 13.921]])
    assert np.array_equal(expected_covariance_matrix, covariance_matrix, equal_nan=True)


def test_constant_regress():
    """
    Verify Constant Regression Model
    """
    tmp = constant_reg.r(np.array([[1, 0, -10]]))
    expected_tmp = np.array([[1.]])
    tmp1 = constant_reg.r(np.array([[1], [0], [-10]]))
    expected_tmp1 = np.array([[1.], [1.], [1.]])
    assert (tmp == expected_tmp).all() and (tmp1 == expected_tmp1).all()


def test_linear_regress():
    """
    Verify Linear Regression Model
    """
    tmp = linear_reg.r(np.array([[1, 0, -10]]))
    expected_tmp = np.array([[1., 1., 0., -10.]])
    tmp1 = linear_reg.r(np.array([[1], [0], [-10]]))
    expected_tmp1 = np.array([[1., 1.], [1., 0.], [1., -10]])
    assert (tmp == expected_tmp).all() and (tmp1 == expected_tmp1).all()


def test_quadratic_regress():
    """
    Verify Quadratic Regression Model
    """
    tmp = quadratic_reg.r(np.array([[1, 2, 3]]))
    expected_tmp = np.array([[1., 1., 2., 3., 1., 2., 3., 4., 6., 9.]])
    tmp1 = quadratic_reg.r(np.array([[1], [0], [-10]]))
    expected_tmp1 = np.array([[1., 1., 1], [1., 0., 0.], [1., -10, 100]])
    assert (tmp == expected_tmp).all() and (tmp1 == expected_tmp1).all()


def test_hyperparameters_length():
    """
        Raises an error if shape/length of hyperparameters is not correct.
    """
    with pytest.raises(RuntimeError):
        gpr5 = GaussianProcessRegression(regression_model=constant_reg, kernel=RBF(), hyperparameters=[2.852, 2.959],
                                         random_state=1, optimizer=optimizer3, bounds=[[0.1, 5], [0.1, 3]], noise=True)
        gpr5.fit(samples=samples, values=values)
#
#
# def test_jacobian():
#     jacobian = np.round(gpr.jacobian([[np.pi], [np.pi/2]]), 3)
#     expected_jacobian = np.array([-0., -1.])
#     assert (expected_jacobian == jacobian).all()
#
#
# def test_jacobian1():
#     jacobian = np.round(krig3.jacobian([[np.pi], [np.pi/2]]), 3)
#     expected_jacobian = np.array([0., -1.])
#     assert (expected_jacobian == jacobian).all()
#
#
# def test_regress():
#     from UQpy.surrogates.kriging.regression_models import Constant, Linear, Quadratic
#     krig.regression_model = Constant()
#     tmp = krig.regression_model.r([[0], [1]])
#     tmp_test1 = (tmp[0] == np.array([[1.], [1.]])).all() and (tmp[1] == np.array([[[0.]], [[0.]]])).all()
#
#     krig.regression_model = Linear()
#     tmp = krig.regression_model.r([[0], [1]])
#     tmp_test2 = (tmp[0] == (np.array([[1., 0.], [1., 1.]]))).all() and \
#                 (tmp[1] == np.array([[[0., 1.]], [[0., 1.]]])).all()
#
#     krig.regression_model = Quadratic()
#     tmp = krig.regression_model.r([[-1, 1], [2, -0.5]])
#     tmp_test3 = (tmp[0] == np.array([[1., -1.,  1.,  1., -1.,  1.], [1.,  2., -0.5,  4., -1.,  0.25]])).all() and \
#                 (tmp[1] == np.array([[[0., 1., 0., -2., 1., 0.],
#                                       [0., 0., 1., 0., -1., 2.]],
#                                      [[0., 1., 0., 4., -0.5, 0.],
#                                       [0., 0., 1., 0., 2., -1.]]])).all()
#
#     assert tmp_test1 and tmp_test2 and tmp_test3
#
#
# def test_corr():
#     from UQpy.surrogates.kriging.correlation_models import Exponential, Linear, Spherical, Cubic, Spline
#     krig.correlation_model = Exponential()
#     rx_exponential = (np.round(krig.correlation_model.c([[0], [1], [2]], [[2]], np.array([1])), 3) ==
#                       np.array([[0.135], [0.368], [1.]])).all()
#     drdt_exponential = (np.round(krig.correlation_model.c([[0], [1], [2]], [[2]], np.array([1]), dt=True)[1], 3) ==
#                         np.array([[[-0.271]], [[-0.368]], [[0.]]])).all()
#     drdx_exponential = (np.round(krig.correlation_model.c([[0], [1], [2]], [[2]], np.array([1]), dx=True)[1], 3) ==
#                         np.array([[[0.135]], [[0.368]], [[0.]]])).all()
#     expon = rx_exponential and drdt_exponential and drdx_exponential
#
#     krig.correlation_model = Linear()
#     rx_linear = (np.round(krig.correlation_model.c([[0], [1], [2]], [[2]], np.array([1])), 3) ==
#                  np.array([[0.], [0.], [1.]])).all()
#     drdt_linear = (np.round(krig.correlation_model.c([[0.4], [0.5], [0.6]], [[0.5]], np.array([1]), dt=True)[1], 3) ==
#                    np.array([[[-0.1]], [[-0.]], [[-0.1]]])).all()
#     drdx_linear = (np.round(krig.correlation_model.c([[0.4], [0.5], [0.6]], [[0.5]], np.array([1]), dx=True)[1], 3) ==
#                    np.array([[[1.]], [[-0.]], [[-1.]]])).all()
#     linear = rx_linear and drdt_linear and drdx_linear
#
#     krig.correlation_model = Spherical()
#     rx_spherical = (np.round(krig.correlation_model.c([[0], [1], [2]], [[2]], np.array([1])), 3) ==
#                     np.array([[0.], [0.], [1.]])).all()
#     drdt_spherical = (np.round(krig.correlation_model.c([[0.4], [0.5], [0.6]], [[0.5]], np.array([1]), dt=True)[1], 3)
#                       == np.array([[[-0.148]], [[-0.]], [[-0.148]]])).all()
#     drdx_spherical = (np.round(krig.correlation_model.c([[0.4], [0.5], [0.6]], [[0.5]], np.array([1]), dx=True)[1], 3)
#                       == np.array([[[1.485]], [[-0.]], [[-1.485]]])).all()
#     spherical = rx_spherical and drdt_spherical and drdx_spherical
#
#     krig.correlation_model = Cubic()
#     rx_cubic = (np.round(krig.correlation_model.c([[0.2], [0.5], [1]], [[0.5]], np.array([1])), 3) ==
#                 np.array([[0.784], [1.], [0.5]])).all()
#     drdt_cubic = (np.round(krig.correlation_model.c([[0.4], [0.5], [0.6]], [[0.5]], np.array([1]), dt=True)[1], 3) ==
#                   np.array([[[-0.054]], [[0.]], [[-0.054]]])).all()
#     drdx_cubic = (np.round(krig.correlation_model.c([[0.4], [0.5], [0.6]], [[0.5]], np.array([1]), dx=True)[1], 3) ==
#                   np.array([[[0.54]], [[0.]], [[-0.54]]])).all()
#     cubic = rx_cubic and drdt_cubic and drdx_cubic
#
#     krig.correlation_model = Spline()
#     rx_spline = (np.round(krig.correlation_model.c([[0.2], [0.5], [1]], [[0.5]], np.array([1])), 3) ==
#                  np.array([[0.429], [1.], [0.156]])).all()
#     drdt_spline = (np.round(krig.correlation_model.c([[0.4], [0.5], [0.6]], [[0.5]], np.array([1]), dt=True)[1], 3) ==
#                    np.array([[[-0.21]], [[0.]], [[-0.21]]])).all()
#     drdx_spline = (np.round(krig.correlation_model.c([[0.4], [0.5], [0.6]], [[0.5]], np.array([1]), dx=True)[1], 3) ==
#                    np.array([[[2.1]], [[0.]], [[-2.1]]])).all()
#     spline = rx_spline and drdt_spline and drdx_spline
#
#     assert expon and linear and spherical and cubic and spline
#
#
# def test_reg_model():
#     """
#         Raises an error if reg_model is not callable or a string of an in-built model.
#     """
#     with pytest.raises(BeartypeCallHintPepParamException):
#         Kriging(regression_model='A', correlation_model=Gaussian(), correlation_model_parameters=[1])
#
#
# def test_corr_model():
#     """
#         Raises an error if corr_model is not callable or a string of an in-built model.
#     """
#     with pytest.raises(BeartypeCallHintPepParamException):
#         Kriging(regression_model=Linear(), correlation_model='A', correlation_model_parameters=[1])
#
#
# def test_corr_model_params():
#     """
#         Raises an error if corr_model_params is not defined.
#     """
#     with pytest.raises(TypeError):
#         Kriging(regression_model=Linear(), correlation_model=Gaussian(), bounds=[[0.01, 5]],
#                 optimizations_number=100, random_state=1)
#
#
# def test_optimizer():
#     """
#         Raises an error if corr_model_params is not defined.
#     """
#     with pytest.raises(BeartypeCallHintPepParamException):
#         Kriging(regression_model=Linear(), correlation_model=Gaussian(),
#                 correlation_model_parameters=[1], optimizer='A')
#
#
# def test_random_state():
#     """
#         Raises an error if type of random_state is not correct.
#     """
#     with pytest.raises(BeartypeCallHintPepParamException):
#         Kriging(regression_model=Linear(), correlation_model=Gaussian(),
#                 correlation_model_parameters=[1], random_state='A')
#
#
# def test_loglikelihood():
#     prediction = np.round(krig3.log_likelihood(np.array([1.5]),
#                                                krig3.correlation_model, np.array([[1], [2]]),
#                                                np.array([[1], [1]]), np.array([[1], [2]]), return_grad=False), 3)
#     expected_prediction = 1.679
#     assert (expected_prediction == prediction).all()
#
#
# def test_loglikelihood_derivative():
#     prediction = np.round(krig3.log_likelihood(np.array([1.5]), krig3.correlation_model, np.array([[1], [2]]),
#                                                np.array([[1], [1]]), np.array([[1], [2]]), return_grad=True)[1], 3)
#     expected_prediction = np.array([-0.235])
#     assert (expected_prediction == prediction).all()
#
# def test_example():
#     from UQpy.surrogates import Kriging
#     from UQpy.utilities.strata import Rectangular
#     from UQpy.sampling import StratifiedSampling, MonteCarloSampling
#     from UQpy.RunModel import RunModel
#     from UQpy.distributions import Uniform
#     import numpy as np
#     import matplotlib.pyplot as plt
#
#     marginals = [Uniform()]
#
#     strata = Rectangular(strata_number=[10])
#
#     x = StratifiedSampling(distributions=marginals, strata_object=strata,
#                            samples_per_stratum_number=1, random_state=2)
#
#     def func(x):
#         return 1 / (1 + (10 * x) ** 4) + 0.5 * np.exp(-100 * (x - 0.5) ** 2)+0.05
#
#     samples = x.samples.copy()
#     output = np.zeros([x.samples.shape[0], 1])
#     for j in range(x.samples.shape[0]):
#         output[j, 0] = func(x.samples[j, 0])
#
#     from UQpy.surrogates.kriging.regression_models import Linear
#     from UQpy.surrogates.kriging.correlation_models import Gaussian
#
#     from UQpy.utilities.optimization.MinimizeOptimizer import MinimizeOptimizer
#     from UQpy.surrogates.kriging.constraints.Nonnegative import Nonnegative
#     optimizer = MinimizeOptimizer(method="cobyla", bounds=[[1, 1000]])
#
#     K = Kriging(regression_model=Linear(), correlation_model=Gaussian(), optimizer=optimizer,
#                 optimizations_number=20, correlation_model_parameters=[1], random_state=2,
#                 optimize_constraints=Nonnegative(np.linspace(min(x.samples), max(x.samples), 30)))
#
#     K.fit(samples=samples, values=output)
#     print(K.correlation_model_parameters)
#
#     num = 1000
#     x1 = np.linspace(min(x.samples), max(x.samples), num)
#
#     y, y_sd = K.predict(x1.reshape([num, 1]), return_std=True)
#     y_grad = K.jacobian(x1.reshape([num, 1]))
#
#     y_act = np.zeros([num, 1])
#     for i in range(num):
#         y_act[i, 0] = func(x1[i, 0])
#
#     fig = plt.figure(figsize=(10, 8))
#     ax = plt.subplot(111)
#     plt.plot(x1, y_act, label='Actual fucntion')
#     plt.plot(x1, y, label='Surrogate')
#     # plt.plot(x1, y_grad, label='Gradient')
#     plt.scatter(K.samples, K.values, label='Data')
#     plt.fill(np.concatenate([x1, x1[::-1]]), np.concatenate([y - 1.9600 * y_sd,
#                                                              (y + 1.9600 * y_sd)[::-1]]),
#              alpha=.5, fc='y', ec='None', label='95% CI')
#     box = ax.get_position()
#     ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#
#     # Put a legend to the right of the current axis
#     ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     plt.grid()
#     plt.show()