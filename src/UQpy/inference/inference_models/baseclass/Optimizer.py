from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):

    @abstractmethod
    def optimize(self, function, initial_guess):
        pass

    def compute_initial_guess(self):
        if x0 is None:
            if not (isinstance(nopt, int) and nopt >= 1):
                raise ValueError('UQpy: nopt should be an integer >= 1.')
            from UQpy.distributions import Uniform
            x0 = Uniform().rvs(
                nsamples=nopt * self.inference_model.parameters_number, random_state=self.random_state).reshape(
                (nopt, self.inference_model.parameters_number))
            if 'bounds' in self.kwargs_optimizer.keys():
                bounds = np.array(self.kwargs_optimizer['bounds'])
                x0 = bounds[:, 0].reshape((1, -1)) + (bounds[:, 1] - bounds[:, 0]).reshape((1, -1)) * x0
        else:
            x0 = np.atleast_2d(x0)
            if x0.shape[1] != self.inference_model.parameters_number:
                raise ValueError('UQpy: Wrong dimensions in x0')

