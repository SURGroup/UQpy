def example1(samples):

    # g(X) = R - S
    R = samples[0]
    S = samples[1]
    return R-S
    

def example2(samples):
    import numpy as np
    d = 2
    beta = 3.0902
    u = -1/np.sqrt(d) * (samples[0] + samples[1]) + beta
    return u


def example3(samples):
    u = 6.2*samples[0] - samples[1]*samples[2]**2
    return u
    
def example4(samples):
    u = samples[0]*samples[1] - 80
    return u