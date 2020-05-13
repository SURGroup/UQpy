def rastrigin(z):
    # A highly non- linear performance function involving non-convex and non-connex domains of failure
    # (i.e. ‘‘scattered gaps of failure’’)
    import numpy as np
    return 10 - (z[0]**2 + z[1]**2 - 5*np.cos(2*np.pi*z[0]) - 5*np.cos(2*np.pi*z[1]))
