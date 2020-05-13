def jenn_sam(z):
    # Jennrich-Sampson's Function
    import numpy as np
    I = np.arange(10)+1
    a = (2 + 2*I - (np.exp(I*z[0]) + np.exp(I*z[1])))**2
    return np.sum(a)
