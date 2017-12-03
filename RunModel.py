from SampleMethods import Strata
from RunModel import *
import numpy as np


class RunModel:

    def __init__(self, generator=None, input=None, nsamples=None, method=None, interpreter=None, model=None, Type=None, \
                 sts_input=None):

        """
        Class of methods used in order to evaluate a model (FEM, PDE, e.t.c)

        param generator: Class of sampling methods

        param input:  This class takes as input the sample points at which the model will be evaluated
                        either as an

         a) np.array([n, dim])
         b).txt input file under the name 'samples.txt'.


         param method:

        If no input is provided then generate the sample points generated with the following methods:

        1. Monte Carlo simulation (mcs)

        2. Latin hypercube sampling (lhs)

        3. Stratified Sampling  (sts)

        4.  Markov Chain Monte Carlo simulation (mcmc)

        :param nsamples: Number of model evaluations. Provided only if input=None

        :param interpreter: The type of interpreter used to run the model. If interpreter=None then python is assumed.

        1. python

        2. matlab

        3. commercial packages (Abaqus, Ansys, e.t.c)

        :param model: Provides the name of the model. If model=None then the program is terminated.

        :param Type: Type of output depending its form

        Options:

        1. scalar

        2. vector

        3. tensor

        If Type=None then a scalar quantity is assumed.

        Created by: Dimitris G. Giovanis
        Last modified: 12/1/2017
        Last modified by: Dimitris G. Giovanis
        """

        is_string = isinstance(input, str)
        if input is None:
            sm = generator
            self.method = method
            self.dimension = sm.dimension
            if self.method == 'mcs':
                self.nsamples = int(nsamples)
                mcs = sm.MCS(self.nsamples, self.dimension)
                self.samples = mcs.samples
            elif self.method == 'lhs':
                self.nsamples = int(nsamples)
                self.samples = sm.lhs()
            elif self.method == 'sts':
                is_string = isinstance(sts_input, str)
                if is_string:
                    ss = sm.STS(strata=Strata(input_file=sts_input))
                else:
                    ss = sm.STS(strata=Strata(nstrata=sts_input))
                self.nsamples = ss.samples.shape[0]
                self.samples = ss.samples
                print()
            elif self.method == 'mcmc':
                self.samples = sm.mcmc(self.nsamples, self.dimension)
        elif is_string:
            self.samples = np.loadtxt(input, dtype=np.float32, delimiter=' ')
            self.nsamples = self.samples.shape[0]
            self.dimension = self.samples.shape[1]
        else:
            self.samples = input
            self.nsamples = self.samples.shape[0]
            self.dimension = self.samples.shape[1]
            self.method = None
            print()

        if model is None:
            raise NotImplementedError('No model is provided')
        else:
            self.model = model

        if interpreter is None:
            self.interpreter = 'python'
        else:
            self.interpreter = interpreter

        if Type is None:
            self.Type = 'scalar'
        else:
            self.Type = Type

        self.eval = self.run_model()

    def run_model(self):

        if self.interpreter == 'python' and self.Type == 'scalar':
            geval = np.zeros(self.nsamples)
            for i in range(self.nsamples):
                geval[i] = self.model(self.samples[i, :], self.Type)

        else:
            raise NotImplementedError('Only python interpreter supported so far and only for scalars')

        return geval












'''

        1. The number of samples to generate (integer)
        2. Whether the random numbers are on [0, 1) or we transform them to the original parameter space
           scale = False:  Keep the random numbers on [0, 1]
           scale = True:  transform to the original space using the inverse of the cumulative distribution
           The default value is False

        If no number is provided then the default number of mc simulations Nmc is set to 100
        The generated samples for all random variables are gather in a list  (len = number of random variables)
        Each element of the list contains the realizations of each random variable (array type with size = Nmc )

        Output: a list containing the realizations of the random variables
            else:
                us = []
                for i in range(self.dimension):
        
                    if self.distribution[i] == 'Uniform':
                        if self.dimension == 1:
                            a = self.parameters[0]
                            b = self.parameters[1]
                        else:
                            a = self.parameters[i][0]
                            b = self.parameters[i][1]
        
                        us.append(a + (b - a) * u[:, i])
        
                    elif self.distribution[i] == 'Normal':
                        a = self.parameters[i][0]
                        b = self.parameters[i][1]
                        us.append(a + (b - a) * stats.norm.ppf(u[:, i]))
        
                    else:
                        raise RuntimeError('distribution not available')
        
                mcs = us
        
            return mcs

'''
