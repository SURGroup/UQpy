import numpy as np
import scipy.stats as stats

def pdf(x, params):
    return params[1]/params[0]*(x/params[0])**(params[1]-1)*np.exp(-(x/params[0])**params[1])

def cdf(x, params):
    return 1-np.exp(-(x/params[0])**params[1])

def icdf(x, params):
    return params[0]*(-np.log(1-x))**(1/params[1])

def log_pdf(x, params):
    return np.log(pdf(x, params))


