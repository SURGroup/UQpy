
import numpy as np

domain = np.linspace(0, 10, 50)


def model_quadratic(theta):
    # this one takes one parameter vector theta and return one qoi
    inpt = np.array(theta).reshape((-1,))
    return inpt[0] * domain + inpt[1] * domain ** 2


def model_quadratic_vectorized(theta):
    # this one takes several parameter vector theta - vectorized computations to return appropriate qois
    inpts = np.tile(theta[..., np.newaxis], [1, 1, 50])
    return inpts[:, 0, :] * domain + inpts[:, 1, :] * domain ** 2