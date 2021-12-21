import numpy as np


def function(z, a=1, b=5.1/(4*np.pi**2), c=5/np.pi, r=6, s=10, t=1/(8*np.pi)):
    f = a*(z[:, 1] - b*z[:, 0]**2 + c*z[:, 0] - r)**2 + s*(1 - t)*np.cos(z[:, 0]) + s + 5*z[:, 0]
    return f