"""

Auxiliary file
==============================================

"""
import numpy as np



def model_quadratic(theta):
    domain = np.linspace(0, 10, 50)
    # this one takes one parameter vector theta and return one qoi
    inpt = np.array(theta).reshape((-1,))
    return inpt[0] * domain + inpt[1] * domain ** 2