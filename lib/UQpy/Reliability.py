import UQpy as uq
import numpy as np
import warnings
import time


def init_rm(data):
    ################################################################################################################
    # Add available sampling methods Here
    valid_methods = ['SuS']

    ################################################################################################################
    # Check if requested method is available

    if 'Method' in data:
        if data['Method'] not in valid_methods:
            raise NotImplementedError("Method - %s not available" % data['Method'])
    else:
        raise NotImplementedError("No reliability method was provided")

    ####################################################################################################################
    # Subset Simulation simulation block.
    # Necessary MCMC parameters:  1. Proposal pdf, 2. Proposal width, 3. Target pdf, 4. Target pdf parameters
    #                             5. algorithm
    # Optional: 1. Seed, 2. skip

    if data['Method'] == 'SuS':
        if 'Probability distribution (pdf)' not in data:
            raise NotImplementedError("Probability distribution not provided")
        if 'Probability distribution parameters' not in data:
            raise NotImplementedError("Probability distribution parameters not provided")
        if 'Names of random variables' not in data:
            raise NotImplementedError('Number of random variables cannot be defined. Specify names of random variables')
        if 'seed' not in data:
            data['seed'] = np.zeros(len(data['Names of random variables']))
        if 'skip' not in data:
            data['skip'] = None
        if 'Proposal distribution' not in data:
            data['Proposal distribution'] = None
        else:
            print(data['Proposal distribution'])
            if data['Proposal distribution'] not in ['Uniform', 'Normal']:
                raise ValueError('Invalid Proposal distribution type. Available distributions: Uniform, Normal')

        if 'Target distribution' not in data:
            data['Target distribution'] = None
        else:
            if data['Target distribution'] not in ['multivariate_pdf', 'marginal_pdf', 'normal_pdf']:
                raise ValueError('InvalidTarget distribution type. Available distributions: multivariate_pdf, '
                                 'marginal_pdf')

        if 'Target distribution parameters' not in data:
            data['Target distribution parameters'] = None

        if 'Proposal distribution width' not in data:
            data['Proposal distribution width'] = None

        if 'MCMC algorithm' not in data:
            data['MCMC algorithm'] = None

        if 'Number of Samples per subset' not in data:
            data['Number of Samples per subset'] = None

        if 'skip' not in data:
            data['skip'] = None

        if 'Conditional probability' not in data:
            data['Conditional probability'] = None

        if 'Limit-state' not in data:
            data['Limit-state'] = None

    ####################################################################################################################
    # Check any NEW RELIABILITY METHOD HERE
    #
    #

    ####################################################################################################################
    # Check any NEW RELIABILITY METHOD HERE
    #
    #


def run_rm(self, data):
    ################################################################################################################
    # Run Subset Simulation
    if data['Method'] == 'SuS':
        print("\nRunning  %k \n", data['Method'])
        sus = SubsetSimulation(self.args, pdf_type=data['Probability distribution (pdf)'],
                               dimension=len(data['Probability distribution (pdf)']),
                               pdf_params=data['Probability distribution parameters'],
                               pdf_target_type=data['Target distribution'],
                               algorithm=data['MCMC algorithm'], pdf_proposal_type=data['Proposal distribution'],
                               pdf_proposal_width=data['Proposal distribution width'],
                               pdf_target_params=data['Target distribution parameters'], seed=data['seed'],
                               skip=data['skip'], nsamples_ss=data['Number of Samples per subset'],
                               p0=data['Conditional probability'], fail=data['Limit-state'])
        return sus


########################################################################################################################
########################################################################################################################
#                                        Subset Simulation (Sus)
########################################################################################################################
class SubsetSimulation:
    """
    A class used to perform Subset Simulation.


    Created by: Dimitris G. Giovanis
    Last modified: 12/08/2017
    Last modified by: Dimitris G. Giovanis

    """

    def __init__(self, args, dimension=None, pdf_type=None, pdf_params=None, pdf_proposal_type=None,
                 pdf_proposal_width=None, pdf_target_type=None,
                 pdf_target_params=None, algorithm=None,   skip=None, seed=None, p0=None,
                 fail=None, nsamples_ss=None):

        self.args = args
        self.nsamples_ss = nsamples_ss
        self.p0 = p0
        self.skip = skip
        self.algorithm = algorithm
        self.pdf_proposal_type = pdf_proposal_type
        self.pdf_target_params = pdf_target_params
        self.pdf_target_type = pdf_target_type
        self.pdf_proposal_width = pdf_proposal_width
        self.limitState = fail
        self.dimension = dimension
        self.seed = seed
        self.pdf_type = pdf_type
        self.pdf_params = pdf_params

        self.init_sus()
        self.pf, self.samples, self.cov = self.run_sus()

        # TODO: DG - Add coefficient of variation estimator for subset simulation

    def run_sus(self):
        step = 0
        y_mcs = np.zeros(self.nsamples_ss)
        p = list()
        cov = list()
        threshold = []
        mcs = UQpyModules.MCS(pdf_type=self.pdf_type, pdf_params=self.pdf_params, nsamples=self.nsamples_ss)
        theta = mcs.samples
        import itertools
        pdf = list(itertools.repeat('Normal', self.dimension))
        params = list(itertools.repeat([0, 1], self.dimension))
        theta_u = PDFs.inv_cdf(mcs.samplesU01, pdf, params)

        print("\nRunning  %k \n", 'MCS')
        for i in range(self.nsamples_ss):
            # Perform MCS
            np.savetxt('UQpy_Samples.txt', theta[i, :], newline=' ', fmt='%0.5f')
            model = UQpyModules.RunModel(self.args)
            y_mcs[i] = model.values

        y, theta_new, m = self.threshold0value(theta_u, y_mcs)
        threshold.append(y)
        prob, di = self.statistical(y_mcs, threshold, step)
        p.append(prob)
        cov.append(di)
        print(y)
        time.sleep(15)

        while threshold[step] < self.limitState:
            step = step + 1
            [theta_newest, y_mcmc] = self.subset_step(self.args, theta_new, threshold[step-1], m)
            ytemp, theta_new, m = self.threshold0value(theta_newest, y_mcmc)
            threshold.append(ytemp)
            prob, di = self.statistical(y_mcmc, threshold, step)
            p.append(prob)
            cov.append(di)
            print(ytemp)
            print(cov)
            time.sleep(15)

        samples = PDFs.normal_to_uniform(theta_new, 0, 1)
        theta_fail = PDFs.inv_cdf(samples, self.pdf_type, self.pdf_params)
        print(cov)
        return np.prod(p), theta_fail, np.sum(cov)

    def subset_step(self, args, theta_new, y, M):
        from UQpyLibraries.SampleMethods import MCMC
        sub_germ_u = theta_new
        sub_germ_g = M
        nr = int(1 / self.p0)
        nchains = int(self.nsamples_ss * self.p0)
        sub_tempu = np.zeros(shape=(nr, self.dimension))
        sub_tempg = np.zeros(nr)
        subset_u = sub_germ_u
        subset_g = sub_germ_g

        for i in range(nchains):
            seed_i = sub_germ_u[i, :]
            val_i = sub_germ_g[i]
            rvs = MCMC(dimension=self.dimension, pdf_target_type=self.pdf_target_type,
                       algorithm=self.algorithm, pdf_proposal_type=self.pdf_proposal_type,
                       pdf_proposal_width=self.pdf_proposal_width,
                       pdf_target_params=self.pdf_target_params, seed=seed_i,
                       skip=self.skip, nsamples=nr)

            # Transform samples from N(0, 1) to U(0, 1) and then to the original space
            samples = rvs.samples
            samples_u = PDFs.normal_to_uniform(rvs.samples, 0, 1)
            theta_u = PDFs.inv_cdf(samples_u, self.pdf_type, self.pdf_params)

            for j in range(nr):
                xu = theta_u[j, :]   # In the original space
                candidate = samples[j, :]   # candidate in N(0, 1)
                np.savetxt('UQpy_Samples.txt', xu, newline=' ', fmt='%0.5f')
                model = UQpyModules.RunModel(args)
                temp_g = model.values
                if temp_g < y:
                    val_i = val_i
                    seed_i = seed_i

                else:
                    val_i = temp_g
                    seed_i = candidate

                sub_tempu[j, :] = seed_i
                sub_tempg[j] = val_i
            subset_u = np.concatenate([subset_u, sub_tempu], axis=0)
            subset_g = np.concatenate([subset_g, sub_tempg], axis=0)

        return subset_u, subset_g

    def threshold0value(self, theta, fmc):

        ncr = int((1 - self.p0) * self.nsamples_ss)
        fmc_sorted = np.sort(fmc)
        j = np.argsort(fmc)
        y = fmc_sorted[ncr]
        m = fmc_sorted[ncr+1:]
        jj = j[ncr+1:]

        return y, theta[jj, :], m

    def statistical(self, G, y, step):
        N = G.size
        if y[step] < self.limitState:

            p = self.p0

            if step == 0:
                di = np.sqrt((1 - p) / (p * N))
            else:
                nc = int(p * N)
                r_zero = p * (1 - p)
                I = np.where(G > y[step])
                index = np.zeros(N)
                index[I] = 1
                indices = np.zeros(shape=(int(N / nc), nc)).astype(int)
                for i in range(int(N / nc)):
                    for j in range(nc):
                        if i == 0:
                            indices[i, j] = j
                        else:
                            indices[i, j] = indices[i-1, j] + nc
                gamma = 0
                rho = np.zeros(int(N/nc)-1)
                for k in range(int(N/nc)-1):
                    z = 0
                    for j in range(int(nc)):
                        for l in range(int(N/nc)-k):
                            z = z + index[indices[l, j]] * index[indices[l + k, j]]

                    rho[k] = (1 / (N - k * nc) * z - p ** 2) / r_zero
                    gamma = gamma + 2 * (1 - k * nc / N) * rho[k]

                di = np.sqrt((1 - p) / (p * N) * (1 + gamma))
        else:
            I = np.array(np.where(G < self.limitState))
            p = I.shape[0] / G.size

            nc = int(p * N)
            r_zero = p * (1 - p)
            I = np.where(G > y[step])
            index = np.zeros(N)
            index[I] = 1
            indices = np.zeros(shape=(int(N / nc), nc)).astype(int)
            for i in range(int(N / nc)):
                for j in range(nc):
                    if i == 0:
                        indices[i, j] = j
                    else:
                        indices[i, j] = indices[i - 1, j] + nc
            gamma = 0
            rho = np.zeros(int(N / nc) - 1)
            for k in range(int(N / nc) - 1):
                z = 0
                for j in range(int(nc)):
                    for l in range(int(N / nc) - k):
                        z = z + index[indices[l, j]] * index[indices[l + k, j]]

                rho[k] = (1 / (N - k * nc) * z - p ** 2) / r_zero
                gamma = gamma + 2 * (1 - k * nc / N) * rho[k]

            di = np.sqrt((1 - p) / (p * N) * (1 + gamma))

        return p, di

    def init_sus(self):
        if self.pdf_type is None:
            raise NotImplementedError("Probability distribution not provided")
        else:
            for i in self.pdf_type:
                if i not in ['Uniform', 'Normal', 'Lognormal', 'Weibull', 'Beta', 'Exponential']:
                    raise NotImplementedError("Supported distributions: 'Uniform', 'Normal', 'Lognormal', 'Weibull', "
                                              "'Beta', 'Exponential' ")
        if self.pdf_params is None:
            raise NotImplementedError("Probability distribution parameters not provided")
        if len(self.pdf_type) != len(self.pdf_params):
            raise NotImplementedError("Incompatible dimensions")
        if self.nsamples_ss is None:
            raise NotImplementedError('Number of samples not defined.')
        if self.seed is None:
            self.seed = np.zeros(self.dimension)
        if self.skip is None:
            self.skip = 1
        if self.pdf_proposal_type is None:
            self.pdf_target_type = 'Uniform'
        if self.pdf_proposal_type not in ['Uniform', 'Normal']:
            raise ValueError('Invalid Proposal distribution type. Available distributions: Uniform, Normal')
        if self.pdf_target_type is None:
            self.pdf_target_type = 'marginal_pdf'
        if self.pdf_target_type not in ['multivariate_pdf', 'marginal_pdf']:
            raise ValueError('InvalidTarget distribution type. Available distributions: multivariate_pdf, marginal_pdf')
        if self.pdf_target_params is None:
            warnings.warn('Target parameters not defined. Default values are  [0, 1]')
            self.pdf_target_params = [0, 1]

        if self.pdf_proposal_width is None:
            warnings.warn('Proposal width not defined. Default value is 2')
            self.pdf_proposal_width = 2

        if self.algorithm is None:
            if self.pdf_target_type is not None:
                if self.pdf_target_type in ['marginal_pdf']:
                    warnings.warn('MCMC algorithm not defined. The MMH will be used')
                    self.algorithm = 'MMH'
                elif self.pdf_target_type in ['multivariate_pdf']:
                    warnings.warn('MCMC algorithm not defined. The MH will be used')
                    self.algorithm = 'MH'
        else:
            if self.algorithm not in ['MH', 'MMH']:
                raise NotImplementedError('Invalid MCMC algorithm. Select from: MH, MMH')

