from UQpyLibraries.SampleMethods import *


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

        def __init__(self, sm=None, rm=None, dimension=None, nsamples_ss=None, p0_cond=None, model=None,
                     mcmc_algorithm=None, pdf_proposal_params=None, pdf_proposal=None, pdf_proposal_width=None,
                     pdf_target=None, mcmc_burnIn=None,
                     fail=None, pdf_target_params=None):

            self.nsamples_ss = nsamples_ss
            self.dimension = dimension
            self.p0_cond = p0_cond
            self.burnIn = mcmc_burnIn
            self.model = model
            self.pdf_proposal_params = pdf_proposal_params
            self.algorithm = mcmc_algorithm
            self.pdf_proposal = pdf_proposal
            self.pdf_target = pdf_target
            self.pdf_proposal_width = pdf_proposal_width
            self.fail = fail
            self.marginal_params = pdf_target_params
            self.pf, self.samples, self.solution = self.run_SuS(rm, sm)


            # TODO: DG - Add coefficient of variation estimator for subset simulation

        def run_SuS(self, rm, sm):
            step = 0
            theta_u = np.zeros(shape=(self.nsamples_per_subset, self.dimension))
            Ymc = np.zeros(self.nsamples_per_subset)
            y = []
            p = []

            for i in range(self.nsamples_per_subset):
                theta_u[i, :] = np.random.randn(self.dimension)
                model = rm.Evaluate(rm, theta_u[i, :].reshape(1, self.dimension))
                Ymc[i] = model.v

            ytemp, thetanew, M = self.threshold0value(theta_u, Ymc)
            y.append(ytemp)

            while y[-1] > self.limitState:

                [thetanewest, Y] = self.subset_step(sm, rm, thetanew, y[-1], M)
                step = step + 1
                ytemp, thetanew, M = self.threshold0value(thetanewest, Y)
                y.append(ytemp)
                p.append(self.Statistical(Y, y[-1]))
                print()

            return np.prod(p), thetanew, M

        def subset_step(self, sm, rm, thetanew, y, M):

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
                mcmc = sm.MCMC(sm, nr, target=self.target, x0=seed_i, MCMC_algorithm=self.method, proposal=self.proposal,
                               params=self.proposal_params, marginal_parameters=self.marginal_params, njump=self.jump)

                for j in range(nr):
                    candidate = mcmc.xi[j, :]
                    check_ = np.array_equal(seed_i.reshape(1, self.dimension), candidate)
                    if check_ is not True:
                        model = rm.Evaluate(rm, candidate.reshape(1, self.dimension))
                        allG = model.v
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



