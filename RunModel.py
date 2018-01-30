from SampleMethods import Strata
from  SampleMethods import *
from RunModel import *
import numpy as np


class RunModel:
    """
    :param model: Model to be evaluated
    :param interpreter: Matlab, Python, Abaques, e.t.c
    :param Type: Scalar, Vector, Output


    Created by: Dimitris G. Giovanis
    Last modified: 1/8/2017
    Last modified by: Dimitris G. Giovanis
    """
    def __init__(self, model=None, interpreter=None, Type=None):

        self.model = model
        self.interpreter = interpreter
        self.Type = Type

    class Evaluate:
        """
        :param points:  Points where the model will be evaluated
        """
        def __init__(self, generator=None, points=None):

            is_string = isinstance(points, str)
            if is_string:
                self.points = np.loadtxt(points, dtype=np.float32, delimiter=' ')
                self.N = self.points.shape[0]
                self.d = self.points.shape[1]
            else:
                self.points = points
                self.N = self.points.shape[0]
                self.d = self.points.shape[1]

            if generator is None:
                raise NotImplementedError('No RunModel generator is provided')

            if generator.model is None:
                raise NotImplementedError('No model is provided')

            if generator.interpreter is None:
                generator.interpreter = 'python'

            if generator.Type is None:
                generator.Type = 'scalar'

            self.v = np.zeros(self.N)
            for i in range(self.N):
                self.v[i] = generator.model(self.points[i, :], generator.Type)

    class Eigenvalue:
        """
        Evaluate the eigenvalues corresponding to SROM approximation and compares with MCS solution
        """
        def __int__(self, generator=None, samples=None, dimension=None, probability=None):

            self.samples = samples
            self.dimension = dimension
            self.probability = probability
            if self.interpreter == 'python' and self.Type == 'scalar':
                srom_eigen = np.zeros([self.nsamples, self.dimension])
                for i in range(self.nsamples):
                    srom_eigen[i] = self.model(self.samples[i, :])

            else:
                raise NotImplementedError('Only python interpreter supported so far and only for scalars')

            n_mcs = 1000
            X_mcs = np.random.gamma(shape=2, scale=3, size=[n_mcs, self.dimension]) + np.ones([n_mcs, self.dimension])
            lm_mcs = np.empty([n_mcs, self.dimension])
            for i in range(n_mcs):
                x = X_mcs[i, :]
                Coeff = [-1, x[0] + 2 * x[1], -(x[0] * x[1] + 2 * x[0] * x[2] + 3 * x[1] * x[2] + x[2] ** 2),
                         (x[0] * x[1] * x[2] + (x[0] + x[1]) * x[2])]
                lm_mcs[i, :] = np.roots(Coeff)

            p = np.transpose(np.matrix(self.probability))
            com = np.append(srom_eigen, p, 1)
            srom_eigen = com[np.argsort(com[:, 0].flatten())]
            self.eigen_srom = srom_eigen
            self.eigen_mcs = lm_mcs