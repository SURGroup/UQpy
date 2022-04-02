from UQpy.surrogates.gaussian_process.kernels.baseclass.Kernel import *
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.special import gamma, kv


class Matern(Kernel):
    def __init__(self, nu=1.5):
        """
        Matern Kernel is a generalization of Radial Basis Function kernel.

        :params nu: Shape parameter. For nu=0.5, 1.5, 2.5 and infinity, matern coincides with the exonential,
         matern-3/2, matern-5/2 and RBF covariance function, respectively.
        """
        self.nu = nu

    def c(self, x, s, params, dt=False, dx=False):
        stack = cdist(x/params, s/params, metric='euclidean')
        if self.nu == 0.5:
            return np.exp(-stack)
        elif self.nu == 1.5:
            return (1+np.sqrt(3)*stack)*np.exp(-np.sqrt(3)*stack)
        elif self.nu == 2.5:
            return (1+np.sqrt(5)*stack+5*(stack**2)/3)*np.exp(-np.sqrt(5)*stack)
        elif self.nu == np.inf:
            return np.exp(-(stack**2)/2)
        else:
            stack *= np.sqrt(2*self.nu)
            tmp = 1/(gamma(self.nu)*(2**(self.nu-1)))
            tmp1 = stack ** self.nu
            tmp2 = kv(tmp1)
            return tmp*tmp1*tmp2