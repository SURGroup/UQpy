def series(z, k=7):
    # A series system with four branches
    import numpy as np
    t1 = 3 + 0.1*(z[:, 0]-z[:, 1])**2-(z[:, 0]+z[:, 1])/np.sqrt(2)
    t2 = 3 + 0.1 * (z[:, 0] - z[:, 1]) ** 2 + (z[:, 0] + z[:, 1]) / np.sqrt(2)
    t3 = z[:, 0] - z[:, 1] + k/np.sqrt(2)
    t4 = z[:, 1] - z[:, 0] + k/np.sqrt(2)
    a = np.minimum(np.minimum(t1, t2), np.minimum(t3, t4))
    return a
