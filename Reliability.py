from collections import Counter
from SampleMethods import *
from RunModel import RunModel


class ReliabilityMethods:
    """
    A class containing methods used to perform reliability analysis

    """

########################################################################################################################
########################################################################################################################
#                                        Subset Simulation (Sus)
########################################################################################################################
    class SubsetSimulation:
        """
        A class used to perform Subset Simulation.

            :param sm:
            :param dimension: Stochastic dimension of the problem
            :param nsamples_per_subset:  number of conditional samples per subset
            :param conditional_prob:  conditional probability of each subset
            :param model: Model to evaluate
            :param mcmc: MCMC algorithm (MH/ MMH)
            :param proposal: Type of proposal distribution
            :param width_of_proposal:  width of proposal distribution
            :param target:  target distribution
            :param pf:  Probability of failure
            :param limit_state: Failure criterion

        Created by: Dimitris G. Giovanis
        Last modified: 12/08/2017
        Last modified by: Dimitris G. Giovanis

        """

        def __init__(self, dimension=None, nsamples_per_subset=None, conditional_prob=None, model=None,
                     MCMC_algorithm=None, params=None, proposal=None, proposal_width=None, target=None,
                     limit_state=None, marginal_params=None):

            self.nsamples_per_subset = nsamples_per_subset
            self.dimension = dimension
            self.p0 = conditional_prob
            self.model = model
            self.params = params
            self.method = MCMC_algorithm
            self.proposal = proposal
            self.target = target
            self.width = proposal_width
            self.limitState = limit_state
            self.marginal_params = marginal_params
            self.pf, self.samples, self.eval = self.run_SuS()
            print()

            # TODO: DG - Add coefficient of variation estimator for subset simulation

        def run_SuS(self):
            step = 0
            theta_u = np.zeros(shape=(self.nsamples_per_subset, self.dimension))
            Ymc = np.zeros(self.nsamples_per_subset)
            y = []
            p = []

            for i in range(self.nsamples_per_subset):
                theta_u[i, :] = np.random.randn(self.dimension)
                model = RunModel(dimension=self.dimension, input=theta_u[i, :].reshape(1, self.dimension),
                                 model=self.model)
                Ymc[i] = model.eval

            ytemp, thetanew, M = self.threshold0value(theta_u, Ymc)
            y.append(ytemp)

            while y[-1] > self.limitState:

                [thetanewest, Y] = self.subset_step(thetanew, y[-1], M)
                step = step + 1
                ytemp, thetanew, M = self.threshold0value(thetanewest, Y)
                y.append(ytemp)
                p.append(self.Statistical(Y, y[-1]))
                print()

            return np.prod(p), thetanew, M

        def subset_step(self, thetanew, y, M):

            subgermU = thetanew
            subgermG = M
            nr = int(1 / self.p0)
            nchains = int(self.nsamples_per_subset * self.p0)
            subtempu = np.zeros(shape=(nr, self.dimension))
            subtempg = np.zeros(nr)
            subsetU = subgermU
            subsetG = subgermG
            for i in range(nchains):
                seed_i = subgermU[i, :]
                val_i = subgermG[i]
                mcmc = SampleMethods.MCMC(nsamples=nr, x0=seed_i, dimension=self.dimension,
                                          MCMC_algorithm=self.method, proposal=self.proposal,
                                          marginal_parameters=self.marginal_params, target=self.target)
                for j in range(nr):
                    candidate = mcmc.samples[j, :]
                    check_ = np.array_equal(seed_i.reshape(1, self.dimension), candidate)
                    if check_ is not True:
                        model = RunModel(dimension=self.dimension, input=candidate.reshape(1, self.dimension),
                                         model=self.model)
                        allG = model.eval
                        if allG > y:
                            val_i = val_i
                            seed_i = seed_i
                        else:
                            val_i = allG
                            seed_i = candidate

                    subtempu[j, :] = seed_i
                    subtempg[j] = val_i
                subsetU = np.concatenate([subsetU, subtempu], axis=0)
                subsetG = np.concatenate([subsetG, subtempg], axis=0)

            return subsetU, subsetG

            return U, G

        def threshold0value(self, theta, fmc):

            ncr = int(self.p0 * self.nsamples_per_subset)
            fmc_sorted = np.sort(fmc, axis=0)
            J = np.argsort(fmc, axis=0)
            y = fmc_sorted[ncr]
            M = fmc_sorted[:ncr]
            JJ = J[:ncr]

            return y, theta[JJ, :], M

        def Statistical(self, G, y):
            if y > self.limitState:

                return self.p0

            else:
                I = np.array(np.where(G < self.limitState))
                p = I.shape[0] / G.size
                return p



