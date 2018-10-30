import numpy as np


def Rosenbrock(x, params):
    return np.exp(-(100*(x[1]-x[0]**2)**2+(1-x[0])**2)/params[0])


