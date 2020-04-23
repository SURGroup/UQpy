# These functions define the methods of the Rosenbrock bivariate distribution, here pdf and log_pdf. 
# input x must be of the correct dimension, i.e., (nsamples, 2).
# params must be a list of parameters

import numpy as np


def pdf(x, params):
    return np.exp(-(100*(x[:, 1]-x[:, 0]**2)**2+(1-x[:, 0])**2)/params[0])


def log_pdf(x, params):
    return -(100*(x[:, 1]-x[:, 0]**2)**2+(1-x[:, 0])**2)/params[0]