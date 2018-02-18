from UQpyLibraries.UQpyModules import *


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

    def __init__(self, args, data):

        self.nsamples_ss = data['Number of Samples per subset']
        self.p0 = data['Conditional probability']
        self.burnIn = data['Burn-in samples']
        self.algorithm = data['MCMC algorithm']
        self.pdf_proposal = data['Proposal distribution']
        self.pdf_target_params = data['Marginal target distribution']
        self.pdf_target = data['Marginal target distribution']
        self.pdf_proposal_width = data['Proposal distribution width']
        self.limitState = data['Limit-state']
        self.names = data['Names of random variables']
        self.dimension = data['Number of random variables']

        args.CPUs_flag = True
        args.ParallelProcessing = False
        self.run_sus(args)

        # TODO: DG - Add coefficient of variation estimator for subset simulation

    def run_sus(self, args):
        step = 0
        theta_u = np.zeros(shape=(self.nsamples_ss, self.dimension))
        y_mcs = np.zeros(self.nsamples_ss)
        y = []
        p = []
        for i in range(self.nsamples_ss):
            theta_u[i, :] = np.random.randn(self.dimension)
            np.savetxt('UQpyOut.txt', theta_u[i, :], newline=' ', fmt='%0.5f')
            x = RunModel(args)
            y_mcs[i] = x.values

        ytemp, theta_new, M = self.threshold0value(theta_u, y_mcs)
        y.append(ytemp)

        while y[-1] > self.limitState:

            [theta_newest, Y] = self.subset_step(args, theta_new, y[-1], M)
            step = step + 1
            ytemp, theta_new, M = self.threshold0value(theta_newest, Y)
            y.append(ytemp)
            p.append(self.statistical(Y, y[-1]))
            print()

        return np.prod(p), theta_new, M

    def subset_step(self, args, theta_new, y, M):
        from UQpyLibraries.SampleMethods import MCMC
        subgermU = theta_new
        subgermG = M
        nr = int(1 / self.p0)
        nchains = int(self.nsamples_ss * self.p0)
        subtempu = np.zeros(shape=(nr, self.dimension))
        subtempg = np.zeros(nr)
        subsetU = subgermU
        subsetG = subgermG
        for i in range(nchains):
            seed_i = subgermU[i, :]
            val_i = subgermG[i]
            rvs = MCMC(dim=self.dimension, pdf_target=None,
                       mcmc_algorithm=self.algorithm, pdf_proposal=self.pdf_proposal,
                       pdf_proposal_width=self.pdf_proposal_width,
                       pdf_target_params=None, mcmc_seed=seed_i,
                       pdf_marg_target_params=self.pdf_target_params,
                       pdf_marg_target=self.pdf_target,
                       mcmc_burnIn=self.burnIn, nsamples=1)

            for j in range(nr):
                candidate = rvs.samples[j, :]
                check_ = np.array_equal(seed_i.reshape(1, self.dimension), candidate)
                if check_ is not True:
                    np.savetxt('UQpyOut.txt', candidate, newline=' ', fmt='%0.5f')
                    x = RunModel(args)
                    allG = x.values
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

        ncr = int(self.p0 * self.nsamples_ss)
        fmc_sorted = np.sort(fmc, axis=0)
        J = np.argsort(fmc, axis=0)
        y = fmc_sorted[ncr]
        M = fmc_sorted[:ncr]
        JJ = J[:ncr]

        return y, theta[JJ, :], M

    def statistical(self, G, y):
        if y > self.limitState:

            return self.p0

        else:
            I = np.array(np.where(G < self.limitState))
            p = I.shape[0] / G.size
            return p




