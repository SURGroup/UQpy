from UQpy.surrogates.gpr.kernels.baseclass.Kernel import *
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.special import gamma, kv


class Matern(Kernel):
    def __init__(self, nu=1.5):
        self.nu = nu

    def c(self, x, s, params):
        stack = cdist(x/params, s/params, metric='euclidean')
        if self.nu == 0.5:
            k = np.exp(-stack)
        elif self.nu == 1.5:
            k = (1+np.sqrt(3)*stack)*np.exp(-np.sqrt(3)*stack)
        elif self.nu == 2.5:
            k = (1+np.sqrt(5)*stack+5*(stack**2)/3)*np.exp(-np.sqrt(5)*stack)
        elif self.nu == np.inf:
            k = np.exp(-(stack**2)/2)
        else:
            stack *= np.sqrt(2*self.nu)
            tmp = 1/(gamma(self.nu)*(2**(self.nu-1)))
            tmp1 = stack ** self.nu
            tmp2 = kv(tmp1)
            k = tmp*tmp1*tmp2
        return k
