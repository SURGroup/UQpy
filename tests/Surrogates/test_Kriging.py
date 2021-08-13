import numpy as np
import pytest
from UQpy.Distributions import Uniform
from UQpy.SampleMethods import MCS
from UQpy.Surrogates import Kriging

dist = Uniform(loc=0, scale=5)
samples = MCS(dist_object=dist, nsamples=20, random_state=0).samples
values = np.cos(samples)
krig = Kriging(reg_model='Linear', corr_model='Gaussian', corr_model_params=[1], bounds=[[0.01, 5]], random_state=1)
krig.fit(samples=samples, values=values, nopt=100, corr_model_params=[1])

krig2 = Kriging(reg_model='Constant', corr_model='Gaussian', corr_model_params=[1], bounds=[[0.01, 5]],
                nopt=100, normalize=False, random_state=2, verbose=True)
krig2.fit(samples=samples, values=values)


# Using the in-built linear regression model as a function
linear_regression_model = Kriging(reg_model='Linear', corr_model='Gaussian', corr_model_params=[1]).reg_model
gaussian_corrleation_model = Kriging(reg_model='Linear', corr_model='Gaussian', corr_model_params=[1]).corr_model

krig3 = Kriging(reg_model=linear_regression_model, corr_model=gaussian_corrleation_model, corr_model_params=[1],
                nopt=100, normalize=False)
krig3.fit(samples=samples, values=values)

# May be solution


def test_fit():
    tmp1 = np.round(krig.corr_model_params, 3) == np.array([1.035])
    tmp2 = np.round(krig2.corr_model_params, 3) == np.array([0.448])
    assert tmp1 and tmp2


def test_predict():
    prediction = np.round(krig.predict([[1], [np.pi/2], [np.pi]], True), 3)
    expected_prediction = np.array([[0.54,  0.0, -1.], [0.004,  0.001,  0.]])
    assert (expected_prediction == prediction).all()


def test_predict1():
    prediction = np.round(krig3.predict([[1], [np.pi/2], [np.pi]]), 3)
    expected_prediction = np.array([[0.54, -0., -1.]])
    assert (expected_prediction == prediction).all()


def test_jacobian():
    jacobian = np.round(krig.jacobian([[np.pi], [np.pi/2]]), 3)
    expected_jacobian = np.array([-0., -1.])
    assert (expected_jacobian == jacobian).all()


def test_jacobian1():
    jacobian = np.round(krig3.jacobian([[np.pi], [np.pi/2]]), 3)
    expected_jacobian = np.array([-0., -0.999])
    assert (expected_jacobian == jacobian).all()


def test_regress():
    krig.reg_model = 'Constant'
    tmp = krig._regress()([[0], [1]])
    tmp_test1 = (tmp[0] == np.array([[1.], [1.]])).all() and (tmp[1] == np.array([[[0.]], [[0.]]])).all()

    krig.reg_model = 'Linear'
    tmp = krig._regress()([[0], [1]])
    tmp_test2 = (tmp[0] == (np.array([[1., 0.], [1., 1.]]))).all() and \
                (tmp[1] == np.array([[[0., 1.]], [[0., 1.]]])).all()

    krig.reg_model = 'Quadratic'
    tmp = krig._regress()([[-1, 1], [2, -0.5]])
    tmp_test3 = (tmp[0] == np.array([[1., -1.,  1.,  1., -1.,  1.], [1.,  2., -0.5,  4., -1.,  0.25]])).all() and \
                (tmp[1] == np.array([[[0., 1., 0., -2., 1., 0.],
                                      [0., 0., 1., 0., -1., 2.]],
                                     [[0., 1., 0., 4., -0.5, 0.],
                                      [0., 0., 1., 0., 2., -1.]]])).all()

    assert tmp_test1 and tmp_test2 and tmp_test3


def test_corr():
    krig.corr_model = 'Exponential'
    rx_exponential = (np.round(krig._corr()([[0], [1], [2]], [[2]], np.array([1])), 3) == np.array([[0.135], [0.368],
                                                                                                    [1.]])).all()
    drdt_exponential = (np.round(krig._corr()([[0], [1], [2]], [[2]], np.array([1]), dt=True)[1], 3) ==
                        np.array([[[-0.271]], [[-0.368]], [[0.]]])).all()
    drdx_exponential = (np.round(krig._corr()([[0], [1], [2]], [[2]], np.array([1]), dx=True)[1], 3) ==
                        np.array([[[0.135]], [[0.368]], [[0.]]])).all()
    expon = rx_exponential and drdt_exponential and drdx_exponential

    krig.corr_model = 'Linear'
    rx_linear = (np.round(krig._corr()([[0], [1], [2]], [[2]], np.array([1])), 3) == np.array([[0.], [0.], [1.]])).all()
    drdt_linear = (np.round(krig._corr()([[0.4], [0.5], [0.6]], [[0.5]], np.array([1]), dt=True)[1], 3) ==
                   np.array([[[-0.1]], [[-0.]], [[-0.1]]])).all()
    drdx_linear = (np.round(krig._corr()([[0.4], [0.5], [0.6]], [[0.5]], np.array([1]), dx=True)[1], 3) ==
                   np.array([[[1.]], [[-0.]], [[-1.]]])).all()
    linear = rx_linear and drdt_linear and drdx_linear

    krig.corr_model = 'Spherical'
    rx_spherical = (np.round(krig._corr()([[0], [1], [2]], [[2]], np.array([1])), 3) ==
                    np.array([[0.], [0.], [1.]])).all()
    drdt_spherical = (np.round(krig._corr()([[0.4], [0.5], [0.6]], [[0.5]], np.array([1]), dt=True)[1], 3) ==
                      np.array([[[-0.148]], [[-0.]], [[-0.148]]])).all()
    drdx_spherical = (np.round(krig._corr()([[0.4], [0.5], [0.6]], [[0.5]], np.array([1]), dx=True)[1], 3) ==
                      np.array([[[1.485]], [[-0.]], [[-1.485]]])).all()
    spherical = rx_spherical and drdt_spherical and drdx_spherical

    krig.corr_model = 'Cubic'
    rx_cubic = (np.round(krig._corr()([[0.2], [0.5], [1]], [[0.5]], np.array([1])), 3) ==
                np.array([[0.784], [1.], [0.5]])).all()
    drdt_cubic = (np.round(krig._corr()([[0.4], [0.5], [0.6]], [[0.5]], np.array([1]), dt=True)[1], 3) ==
                  np.array([[[-0.054]], [[0.]], [[-0.054]]])).all()
    drdx_cubic = (np.round(krig._corr()([[0.4], [0.5], [0.6]], [[0.5]], np.array([1]), dx=True)[1], 3) ==
                  np.array([[[0.54]], [[0.]], [[-0.54]]])).all()
    cubic = rx_cubic and drdt_cubic and drdx_cubic

    krig.corr_model = 'Spline'
    rx_spline = (np.round(krig._corr()([[0.2], [0.5], [1]], [[0.5]], np.array([1])), 3) ==
                 np.array([[0.429], [1.], [0.156]])).all()
    drdt_spline = (np.round(krig._corr()([[0.4], [0.5], [0.6]], [[0.5]], np.array([1]), dt=True)[1], 3) ==
                   np.array([[[-0.21]], [[0.]], [[-0.21]]])).all()
    drdx_spline = (np.round(krig._corr()([[0.4], [0.5], [0.6]], [[0.5]], np.array([1]), dx=True)[1], 3) ==
                   np.array([[[2.1]], [[0.]], [[-2.1]]])).all()
    spline = rx_spline and drdt_spline and drdx_spline

    assert expon and linear and spherical and cubic and spline


def test_reg_model():
    """
        Raises an error if reg_model is not callable or a string of an in-built model.
    """
    with pytest.raises(NotImplementedError):
        Kriging(reg_model='A', corr_model='Gaussian', corr_model_params=[1])


def test_corr_model():
    """
        Raises an error if corr_model is not callable or a string of an in-built model.
    """
    with pytest.raises(NotImplementedError):
        Kriging(reg_model='Linear', corr_model='A', corr_model_params=[1])


def test_corr_model_params():
    """
        Raises an error if corr_model_params is not defined.
    """
    with pytest.raises(NotImplementedError):
        Kriging(reg_model='Linear', corr_model='Gaussian', bounds=[[0.01, 5]], nopt=100, random_state=1)


def test_optimizer():
    """
        Raises an error if corr_model_params is not defined.
    """
    with pytest.raises(TypeError):
        Kriging(reg_model='Linear', corr_model='Gaussian', corr_model_params=[1], optimizer='A')


def test_random_state():
    """
        Raises an error if type of random_state is not correct.
    """
    with pytest.raises(TypeError):
        Kriging(reg_model='Linear', corr_model='Gaussian', corr_model_params=[1], random_state='A')


def test_mle_failure():
    """
        Maximum likelihood estimator failed: Choose different starting point or increase nopt
    """
    with pytest.raises(NotImplementedError):
        krig4 = Kriging(reg_model=linear_regression_model, corr_model=gaussian_corrleation_model, corr_model_params=[1],
                        normalize=False)
        krig4.fit(samples=samples, values=values)
