from SampleMethods import Strata
from RunModel import *
import numpy as np


class RunModel:

    def __init__(self, generator=None, input=None, nsamples=None, method=None, interpreter=None, model=None, Type=None, \
                 sts_input=None, lhs_criterion='random', MCMC_algorithm='MH', proposal=None, target=None, pss_design=None, pss_stratum=None, \
                 x0=None, params=None, jump=None):

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

        4. Partially stratified sampling (pss)

        5.  Markov Chain Monte Carlo simulation (mcmc)

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
                lhs = sm.LHS(self.dimension, self.nsamples, criterion=lhs_criterion)
                self.samples = lhs.samples
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
                self.algorithm = MCMC_algorithm
                self.target = target
                self.nsamples = int(nsamples)
                self.x0 = x0
                self.params = params
                self.jump = jump
                self.proposal = proposal
                mcmc = sm.MCMC(nsamples=self.nsamples, target=self.target, x0=self.x0, MCMC_algorithm=self.algorithm, proposal=self.proposal, params = self.params, njump= self.jump)
                self.samples = mcmc.samples
            elif self.method == 'pss':
                self.pss_design = pss_design
                self.pss_stratum = pss_stratum
                self.nsamples = max(pss_stratum)
                pss = sm.PSS(pss_design=self.pss_design, pss_stratum=self.pss_stratum)
                self.samples = pss.samples
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

