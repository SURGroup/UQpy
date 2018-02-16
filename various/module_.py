from functools import partial
from various.modelist import *
import sys


class README:

    def __init__(self, input_file=None):
        """
        :param input_file:
        """
        self.filename = input_file
        self.data = self.readfile()
        self.check = self.error_checks()

    def readfile(self):
        lines_ = []
        mydict = {}
        count = -1
        for line in open(self.filename):
            rec = line.strip()
            count = count + 1
            if rec.startswith('#'):
                lines_.append(count)

        f = open(self.filename)
        lines = f.readlines()

        for i in range(len(lines_)):
            title = lines[lines_[i]][1:-1]
            # General parameters
            if title == 'Method':
                mydict[title] = lines[lines_[i]+1][:-1]
                print()
            elif title == 'Stochastic dimension':
                mydict[title] = int(lines[lines_[i] + 1][:-1])
            elif title == 'Probability distribution (pdf)':
                dist = []
                for k in range(mydict['Stochastic dimension']):
                    dist.append(lines[lines_[i]+k+1][:-1])
                mydict[title] = dist
            elif title == 'Probability distribution parameters':
                params = []
                for k in range(mydict['Stochastic dimension']):
                    params.append([np.float32(lines[lines_[i]+k+1][0]), np.float32(lines[lines_[i]+k+1][2])])
                mydict[title] = params
            elif title == 'Model':
                mydict[title] = lines[lines_[i] + 1][:-1]
            elif title == 'Number of Samples':
                mydict[title] = int(lines[lines_[i] + 1][:-1])
            # Latin Hypercube parameters
            elif title == 'LHS criterion':
                mydict[title] = lines[lines_[i] + 1][:-1]
            elif title == 'distance metric':
                mydict[title] = lines[lines_[i] + 1][:-1]
            elif title == 'iterations':
                mydict[title] = lines[lines_[i] + 1][:-1]
            # partially stratified sampling
            elif title == 'PSS design':
                pss_design = []
                for k in range(mydict['Stochastic dimension']):
                    pss_design.append(int(lines[lines_[i]+k+1][0]))
                mydict[title] = pss_design
            elif title == 'PSS strata':
                pss_strata = []
                for k in range(mydict['Stochastic dimension']):
                    pss_strata.append(int(lines[lines_[i]+k+1][:-1]))
                mydict[title] = pss_strata
            # stratified sampling
            elif title == 'STS design':
                sts_design = []
                for k in range(mydict['Stochastic dimension']):
                    sts_design.append(int(lines[lines_[i]+k+1][:-1]))
                mydict[title] = sts_design
            # Markov Chain Monte Carlo simulation
            elif title == 'MCMC algorithm':
                mydict[title] = lines[lines_[i] + 1][:-1]
            elif title == 'Proposal distribution':
                mydict[title] = lines[lines_[i] + 1][:-1]
            elif title == 'Proposal distribution parameters':
                proposal_params = []
                for k in range(mydict['Stochastic dimension']):
                    proposal_params.append(np.float32(lines[lines_[i]+k+1][0]))
                mydict[title] = proposal_params
            elif title == 'Target distribution':
                mydict[title] = lines[lines_[i] + 1][:-1]
            elif title == 'Burn-in samples':
                mydict[title] = int(lines[lines_[i] + 1][:-1])
            elif title == 'Marginal target distribution parameters':
                marg_params = []
                for k in range(mydict['Stochastic dimension']):
                    marg_params.append([np.float32(lines[lines_[i]+k+1][0]), np.float32(lines[lines_[i]+k+1][2])])
                mydict[title] = marg_params
            # Subset Simulation
            elif title == 'Number of Samples per subset':
                mydict[title] = int(lines[lines_[i] + 1][:-1])
            elif title == 'Width of proposal distribution':
                mydict[title] = np.float32(lines[lines_[i] + 1][:-1])
            elif title == 'Conditional probability':
                mydict[title] = np.float32(lines[lines_[i] + 1][:-1])
            elif title == 'Failure criterion':
                mydict[title] = np.float32(lines[lines_[i] + 1][:-1])
            # Stochastic Reduced Order Models (SROM)

        return mydict

    def error_checks(self):
        if self.data['Method'] not in ['mcs', 'lhs', 'mcmc', 'pss', 'sts', 'SuS']:
            raise NotImplementedError('Available sampling methods: 1. Monte Carlo (mcs), 2. Latin hypercube(lhs), '
                     '3. Markov chain Monte Carlo (mcmc), 4. Partially stratified (pss), 5. Stratified (sts).'
                     'Available reliability methods: 1. Subset simulation (Sus).')

        if self.data['Method'] == 'mcs':
            if len(self.data['Probability distribution (pdf)']) != self.data['Stochastic dimension']:
                sys.exit('Error: Incompatible dimensions.')

        elif self.data['Method'] == 'lhs':
            if len(self.data['Probability distribution (pdf)']) != self.data['Stochastic dimension']:
                sys.exit('Error: Incompatible dimensions.')

            if 'LHS criterion' not in self.data:
                self.data['LHS criterion'] = 'random'
            elif self.data['LHS criterion'] not in ['random', 'centered', 'maximin', 'correlate', 'correlate_cond']:
                sys.exit('Invalid LHS criterion requested.')

            if 'distance metric' not in self.data:
                self.data['distance metric'] = 'euclidean'
            elif self.data['distance metric'] not in ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine',
                                       'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching',
                                       'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
                                       'sokalsneath', 'sqeuclidean', 'yule']:
                sys.exit('Invalid LHS distance metric requested.')

            if 'iterations' not in self.data:
                self.data['iterations'] = 1000

        elif self.data['Method'] == 'pss':
            if self.data['PSS design'] not in self.data:
                sys.exit('PSS design required.')
            else:
                if len(self.data['PSS design']) != self.data['Stochastic dimension']:
                    sys.exit('Error: Incompatible dimensions.')

            if self.data['PSS strata'] not in self.data:
                sys.exit('PSS strata required.')
            else:
                if len(self.data['PSS strata']) != self.data['Stochastic dimension']:
                    sys.exit('Error: Incompatible dimensions.')

        elif self.data['Method'] == 'sts':
            if self.data['STS design'] not in self.data:
                sys.exit('STS design required.')
            else:
                if len(self.data['STS design']) != self.data['Stochastic dimension']:
                    sys.exit('Error: Incompatible dimensions.')

        elif self.data['Method'] == 'mcmc' or self.data['Method'] == 'SuS':
            if 'MCMC algorithm' not in self.data:
                sys.exit('MCMC algorthm required')
            else:
                if self.data['MCMC algorithm'] not in ['MH', 'MMH']:
                    sys.exit('Algorithm not applicable. Select one of: 1. MH, 2. MMH')

            if 'Proposal distribution' not in self.data:
                sys.exit('Proposal distribution required')
            else:
                if self.data['Proposal distribution'] not in ['Normal', 'Uniform']:
                    sys.exit('Algorithm not applicable. Select one of: 1. Normal, 2. Uniform')

            if 'Target distribution' not in self.data:
                sys.exit('Target distribution required')
            else:
                print()
                # TODO: DG - We need to have a library of available  target distributions
            if 'Marginal target distribution parameters' not in self.data:
                sys.exit('Define marginal distribution parameters')
            else:
                if len(self.data['Marginal target distribution parameters']) != self.data['Stochastic dimension']:
                    sys.exit('Error: Incompatible dimensions.')

            if 'Burn-in samples' not in self.data:
                self.data['Burn-in samples'] = 1

        elif self.data['Method'] == 'SuS':
            if 'Number of Samples per subset' not in self.data:
                sys.exit('Define number of samples per subset')

            if 'Conditional probability' not in self.data:
                raise ValueError('Define conditional probability. Default will be set to 0.1')
                mydict['Conditional probability'] = 0.1
            else:
                if self.data['Conditional probability'] >= 1.0:
                    sys.exit('Conditional Probability must be lower of 1.0')

            if 'Width of proposal distribution' not in self.data:
                raise ValueError('Define width of proposal distribution. Default will be set to 2.')
                mydict['Width of proposal distribution'] = 2.0

            if 'Failure criterion' not in self.data:
                sys.exit('Define failure criterion')

            if 'Model' not in self.data:
                sys.exit('A numerical model is required')

        return 'OK'


def def_model(_model):
    if _model == 'model_zabaras':
        model = partial(model_zabaras)
    elif _model == 'model_ko2d':
        model = partial(model_ko2d)
    elif _model == 'model_reliability':
        model = partial(model_reliability)

    return model


def def_target(target):
    if target == 'mvnpdf':
        target = partial(mvnpdf)
    elif target == 'normpdf':
        target = partial(normpdf)
    elif target == 'marginal':
        target = partial(marginal)
    return target


