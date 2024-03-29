import numpy as np

domain = np.linspace(0, 10, 4)

def model_quadratic1(theta):
    # this one takes one parameter vector theta and return one qoi
    domain = np.linspace(0, 10, 50)
    inpt = np.array(theta).reshape((-1,))
    return inpt[0] * domain + inpt[1] * domain ** 2

def model_quadratic(theta):
    # this one takes one parameter vector theta and return one qoi
    inpt = np.array(theta).reshape((-1,))
    return inpt[0] * domain + inpt[1] * domain ** 2


def model_quadratic_vectorized(theta):
    # this one takes several parameter vector theta - vectorized computations to return appropriate qois
    inpts = np.tile(theta[..., np.newaxis], [1, 1, 50])
    return inpts[:, 0, :] * domain + inpts[:, 1, :] * domain ** 2


def model_linear(theta):
    # this one takes one parameter vector theta and return one qoi
    inpt = np.array(theta).reshape((-1,))
    return inpt[0] * domain


def model_cubic(theta):
    # this one takes one parameter vector theta and return one qoi
    inpt = np.array(theta).reshape((-1,))
    return inpt[0] * domain + inpt[1] * domain ** 2 + inpt[2] * domain ** 3