def y_func(z):
    return 1/(6.2727*(abs(0.3-z[:, 0]**2-z[:, 1]**2)+0.01))
    # return np.sqrt(z[:, 0]**2+z[:, 1]**2)

