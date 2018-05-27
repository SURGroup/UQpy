from UQpy import UQpyModules, Distributions
from UQpy.RunModel import RunModel
from UQpy.SampleMethods import MCMC
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
#                                        Subset Simulation
########################################################################################################################
class SubsetSimulation:
    """
    A class used to perform Subset Simulation.

    This class estimates probability of failure for a user-defined model using Subset Simulation

    References:
    S.-K. Au and J. L. Beck, “Estimation of small failure probabilities in high dimensions by subset simulation,”
        Probabilistic Eng. Mech., vol. 16, no. 4, pp. 263–277, Oct. 2001.

    Input:
    :param dimension:  A scalar value defining the dimension of target density function.
                    Default: 1
    :type dimension: int

    :param nsamples_ss: Number of samples to generate in each conditional subset
                        No Default Value: nsamples_ss must be prescribed
    :type nsamples_ss: int

    :param p_cond: Conditional probability at each level
                        Default: p_cond = 0.1
    :type p_cond: float

    :param algorithm:  Algorithm used to generate MCMC samples.
                    Options:
                        'MH': Metropolis Hastings Algorithm
                        'MMH': Component-wise Modified Metropolis Hastings Algorithm
                        'Stretch': Affine Invariant Ensemble MCMC with stretch moves
                    Default: 'MMH'
    :type algorithm: str

    :param pdf_target_type: Type of target density function for acceptance/rejection in MMH. Not used for MH or Stretch.
                    Options:
                        'marginal_pdf': Check acceptance/rejection for a candidate in MMH using the marginal pdf
                                        For independent variables only
                        'joint_pdf': Check acceptance/rejection for a candidate in MMH using the joint pdf
                    Default: 'marginal_pdf'
    :type pdf_target_type: str

    :param pdf_target: Target density function from which to draw random samples
                    The target joint probability density must be a function, or list of functions, or a string.
                    If type == 'str'
                        The assigned string must refer to a custom pdf defined in the file custom_pdf.py in the working
                            directory
                    If type == function
                        The function must be defined in the python script calling MCMC
                    If dimension > 1 and pdf_target_type='marginal_pdf', the input to pdf_target is a list of size
                        [dimensions x 1] where each item of the list defines a marginal pdf.
                    Default: Multivariate normal distribution having zero mean and unit standard deviation
    :type pdf_target: function, function list, or str

    :param pdf_target_params: Parameters of the target pdf
    :type pdf_target_params: list

    :param pdf_proposal_type: Type of proposal density function for MCMC. Only used with algorithm = 'MH' or 'MMH'
                    Options:
                        'Normal' : Normal proposal density
                        'Uniform' : Uniform proposal density
                    Default: 'Uniform'
                    If dimension > 1 and algorithm = 'MMH', this may be input as a list to assign different proposal
                        densities to each dimension. Example pdf_proposal_type = ['Normal','Uniform'].
                    If dimension > 1, algorithm = 'MMH' and this is input as a string, the proposal densities for all
                        dimensions are set equal to the assigned proposal type.
    :type pdf_proposal_type: str or str list

    :param pdf_proposal_scale: Scale of the proposal distribution
                    If algorithm == 'MH' or 'MMH'
                        For pdf_proposal_type = 'Uniform'
                            Proposal is Uniform in [x-pdf_proposal_scale/2, x+pdf_proposal_scale/2]
                        For pdf_proposal_type = 'Normal'
                            Proposal is Normal with standard deviation equal to pdf_proposal_scale
                    If algorithm == 'Stretch'
                        pdf_proposal_scale sets the scale of the stretch density
                            g(z) = 1/sqrt(z) for z in [1/pdf_proposal_scale, pdf_proposal_scale]
                    Default value: dimension x 1 list of ones
    :type pdf_proposal_scale: float or float list
                    If dimension > 1, this may be defined as float or float list
                        If input as float, pdf_proposal_scale is assigned to all dimensions
                        If input as float list, each element is assigned to the corresponding dimension

    :param model_type: Define the model as a python file or as a third party software model (e.g. Matlab, Abaqus, etc.)
            Options: None - Run a third party software model
                     'python' - Run a python model. When selected, the python file must contain a class RunPythonModel
                                that takes, as input, samples and dimension and returns quantity of interest (qoi) in
                                in list form where there is one item in the list per sample. Each item in the qoi list
                                may take type the user prefers.
            Default: None
    :type model_type: str

    :param model_script: Defines the script (must be either a shell script (.sh) or a python script (.py)) used to call
                            the model.
                         This is a user-defined script that must be provided.
                         If model_type = 'python', this must be a python script (.py) having a specified class
                            structure. Details on this structure can be found in the UQpy documentation.
    :type: model_script: str

    :param input_script: Defines the script (must be either a shell script (.sh) or a python script (.py)) that takes
                            samples generated by UQpy from the sample file generated by UQpy (UQpy_run_{0}.txt) and
                            imports them into a usable input file for the third party solver. Details on
                            UQpy_run_{0}.txt can be found in the UQpy documentation.
                         If model_type = None, this is a user-defined script that the user must provide.
                         If model_type = 'python', this is not used.
    :type: input_script: str

    :param output_script: (Optional) Defines the script (must be either a shell script (.sh) or python script (.py))
                            that extracts quantities of interest from third-party output files and saves them to a file
                            (UQpy_eval_{}.txt) that can be read for postprocessing and adaptive sampling methods by
                            UQpy.
                          If model_type = None, this is an optional user-defined script. If not provided, all run files
                            and output files will be saved in the folder 'UQpyOut' placed in the current working
                            directory. If provided, the text files UQpy_eval_{}.txt are placed in this directory and all
                            other files are deleted.
                          If model_type = 'python', this is not used.
    :type output_script: str

    Output:

    :param pf: Probability of failure estimate
    :type pf: float
    """

    # Created by: Dimitris G.Giovanis, Michael D. Shields
    # Modified: 12 / 08 / 2017
    # Modified by: Dimitris G. Giovanis
    # Modified: 5 / 15 / 2018
    # Modified by: Michael D. Shields

    def __init__(self, dimension=None, samples_init=None, nsamples_ss=None, p_cond=None, pdf_target_type=None,
                 pdf_target=None, pdf_target_params=None, pdf_proposal_type=None, pdf_proposal_scale=None,
                 algorithm=None, model_type=None, model_script=None, input_script=None, output_script=None):

        self.dimension = dimension
        self.samples_init = samples_init
        self.pdf_target_type = pdf_target_type
        self.pdf_target = pdf_target
        self.pdf_target_params = pdf_target_params
        self.pdf_proposal_type = pdf_proposal_type
        self.pdf_proposal_scale = pdf_proposal_scale
        self.nsamples_ss = nsamples_ss
        self.algorithm = algorithm
        self.model_type = model_type
        self.model_script = model_script
        self.input_script = input_script
        self.output_script = output_script
        self.p_cond = p_cond
        self.g = list()
        self.samples = list()
        self.g_level = list()
        self.pf = np.empty(1)

        # Hard-wire the maximum number of conditional levels.
        self.max_level = 20

        # Initialize variables and perform error checks
        self.init_sus()

        # Select the appropriate Subset Simulation Algorithm
        if self.algorithm == 'MMH':
            self.run_subsim_mmh()
        elif self.algorithm == 'Stretch':
            self.run_subsim_stretch()

        # TODO: DG - Add coefficient of variation estimator for subset simulation

    def run_subsim_mmh(self):
        step = 0
        n_keep = int(self.p_cond * self.nsamples_ss)

        # Generate the initial samples - Level 0
        if self.samples_init is None:
            x_init = MCMC(dimension=self.dimension, pdf_proposal_type=self.pdf_proposal_type,
                          pdf_proposal_scale=self.pdf_proposal_scale, pdf_target_type=self.pdf_target_type,
                          pdf_target = self.pdf_target, pdf_target_params = self.pdf_target_params,
                          algorithm=self.algorithm, nsamples=self.nsamples_ss, seed=np.zeros((self.dimension)))
            self.samples.append(x_init.samples)
        else:
            self.samples.append(self.samples_init)

        g_init = RunModel(samples=self.samples[step], model_type=self.model_type, model_script=self.model_script,
                          input_script=self.input_script, output_script=self.output_script, dimension=self.dimension)

        self.g.append(np.asarray(g_init.model_eval.QOI))
        g_ind = np.argsort(self.g[step])
        self.g_level.append(self.g[step][g_ind[n_keep]])

        while self.g_level[step] > 0 and step < self.max_level:

            step = step + 1
            self.samples.append(self.samples[step - 1][g_ind[0:n_keep]])
            self.g.append(self.g[step - 1][g_ind[:n_keep]])

            for i in range(self.nsamples_ss-n_keep):
                seed = self.samples[step][i]

                x_mcmc = MCMC(dimension=self.dimension, pdf_proposal_type=self.pdf_proposal_type,
                              pdf_proposal_scale=self.pdf_proposal_scale, pdf_target_type=self.pdf_target_type,
                              pdf_target = self.pdf_target, pdf_target_params = self.pdf_target_params,
                              algorithm=self.algorithm, nsamples=2, seed=seed)

                x_temp = x_mcmc.samples[1].reshape((1, self.dimension))
                g_model = RunModel(samples = x_temp, cpu=1, model_type=self.model_type, model_script=self.model_script,
                                   input_script=self.input_script, output_script=self.output_script,
                                   dimension=self.dimension)

                g_temp = g_model.model_eval.QOI

                # Accept or reject the sample
                if g_temp < self.g_level[step - 1]:
                    self.samples[step] = np.vstack((self.samples[step], x_temp))
                    self.g[step] = np.hstack((self.g[step],g_temp[0]))
                else:
                    self.samples[step] = np.vstack((self.samples[step], self.samples[step][i]))
                    self.g[step] = np.hstack((self.g[step], self.g[step][i]))

            g_ind = np.argsort(self.g[step])
            self.g_level.append(self.g[step][g_ind[n_keep]])

        n_fail = len([value for value in self.g[step] if value < 0])
        self.pf = self.p_cond**step*n_fail/self.nsamples_ss
        print(self.pf)

    def run_subsim_stretch(self):
        step = 0
        n_keep = int(self.p_cond * self.nsamples_ss)

        # Generate the initial samples - Level 0
        if self.samples_init is None:
            x_init = MCMC(dimension=self.dimension, pdf_proposal_type=self.pdf_proposal_type,
                          pdf_proposal_scale=self.pdf_proposal_scale, pdf_target_type=self.pdf_target_type,
                          pdf_target=self.pdf_target, pdf_target_params=self.pdf_target_params,
                          algorithm='MMH', nsamples=self.nsamples_ss, seed=np.zeros((self.dimension)))
            self.samples.append(x_init.samples)
        else:
            self.samples.append(self.samples_init)

        g_init = RunModel(samples=self.samples[step], model_type=self.model_type, model_script=self.model_script,
                          input_script=self.input_script, output_script=self.output_script,
                          dimension=self.dimension)

        self.g.append(np.asarray(g_init.model_eval.QOI))
        g_ind = np.argsort(self.g[step])
        self.g_level.append(self.g[step][g_ind[n_keep]])

        while self.g_level[step] > 0:

            step = step + 1
            self.samples.append(self.samples[step - 1][g_ind[0:n_keep]])
            self.g.append(self.g[step - 1][g_ind[:n_keep]])

            for i in range(self.nsamples_ss - n_keep):
                seed = self.samples[step][i:i+n_keep]

                x_mcmc = MCMC(dimension=self.dimension, pdf_proposal_type=self.pdf_proposal_type,
                              pdf_proposal_scale=self.pdf_proposal_scale, pdf_target_type=self.pdf_target_type,
                              pdf_target=self.pdf_target, pdf_target_params=self.pdf_target_params,
                              algorithm=self.algorithm, nsamples=n_keep+1, seed=seed)

                x_temp = x_mcmc.samples[n_keep].reshape((1, self.dimension))
                g_model = RunModel(samples=x_temp, cpu=1, model_type=self.model_type,
                                   model_script=self.model_script,
                                   input_script=self.input_script, output_script=self.output_script,
                                   dimension=self.dimension)

                g_temp = g_model.model_eval.QOI

                # Accept or reject the sample
                if g_temp < self.g_level[step - 1]:
                    self.samples[step] = np.vstack((self.samples[step], x_temp))
                    self.g[step] = np.hstack((self.g[step], g_temp[0]))
                else:
                    self.samples[step] = np.vstack((self.samples[step], self.samples[step][i]))
                    self.g[step] = np.hstack((self.g[step], self.g[step][i]))

            g_ind = np.argsort(self.g[step])
            self.g_level.append(self.g[step][g_ind[n_keep]])

        n_fail = len([value for value in self.g[step] if value < 0])
        self.pf = self.p_cond ** step * n_fail / self.nsamples_ss
        print(self.pf)

    def init_sus(self):

        # Set default dimension to 1
        if self.dimension is None:
            self.dimension = 1

        # Check that the number of samples per subset is defined.
        if self.nsamples_ss is None:
            raise NotImplementedError('Number of samples per subset not defined. This is required.')

        # Check that the MCMC algorithm is properly defined.
        if self.algorithm is None:
            self.algorithm = 'MMH'
        elif self.algorithm not in ['Stretch', 'MMH']:
            raise NotImplementedError('Invalid MCMC algorithm. Select from: MMH, Stretch')

        # Check that a valid conditional probability is specified.
        if type(self.p_cond).__name__ != 'float':
            raise NotImplementedError('Invalid conditional probability. p_cond must be of float type.')
        elif self.p_cond <= 0. or self.p_cond >= 1.:
            raise NotImplementedError('Invalid conditional probability. p_cond must be in (0, 1).')

        # Check that the user has defined a model
        if self.model_script is None:
            raise NotImplementedError('Subset Simulation requires the specification of a computational model. Please '
                                      'specify the model using the model_script input.')
