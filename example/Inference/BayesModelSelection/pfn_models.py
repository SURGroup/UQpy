
import numpy as np

domain = np.linspace(0, 10, 122)

def model_linear(inputs=None):
    inputs = np.array(inputs).reshape((-1,))
    return inputs[0]*domain

def model_quadratic(inputs=None):
    inputs = np.array(inputs).reshape((-1,))
    return inputs[0]*domain+inputs[1]*domain**2

def model_cubic(inputs=None):
    inputs = np.array(inputs).reshape((-1,))
    return inputs[0]*domain+inputs[1]*domain**2+inputs[2]*domain**3