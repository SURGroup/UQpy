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


