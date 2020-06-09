import numpy as np

def example1(samples=None):
    g = np.zeros(samples.shape[0])
    for i in range(samples.shape[0]):         
        R = samples[i, 0]
        S = samples[i, 1]
        g[i] = R - S
    return g
    

def example2(samples=None):
    import numpy as np
    d = 2
    beta = 3.0902
    g = np.zeros(samples.shape[0])
    for i in range(samples.shape[0]):
        g[i] = -1/np.sqrt(d) * (samples[i, 0] + samples[i, 1]) + beta
    return g


def example3(samples=None):
    g = np.zeros(samples.shape[0])
    for i in range(samples.shape[0]):
        g[i] = 6.2*samples[i, 0] - samples[i, 1]*samples[i, 2]**2
    return g
    
def example4(samples=None):
    g = np.zeros(samples.shape[0])
    for i in range(samples.shape[0]):
        g[i] = samples[i, 0]*samples[i, 1] - 80
    return g