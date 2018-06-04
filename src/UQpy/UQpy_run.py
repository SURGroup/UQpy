from argparse import ArgumentParser
from UQpy.RunModel import RunModel
from UQpy.SampleMethods import *
from UQpy.Reliability import *
import numpy as np


def readfile(filename):
    lines_ = []
    mydict = {}
    count = -1
    for line in open(filename):
        rec = line.strip()
        count = count + 1
        if rec.startswith('#'):
            lines_.append(count)

    f = open(filename)
    lines = f.readlines()

    for i in range(len(lines_)):
        title = lines[lines_[i]][1:-1]
        ################################################################################################################
        # General parameters
        if title == 'method':
            mydict[title] = lines[lines_[i]+1][:-1]
            print()
        elif title == 'distribution type':
            dist = []
            j = 0
            while j >= 0:
                testline = lines[lines_[i] + j + 1].strip()
                if not testline:
                    break
                else:
                    x = lines[lines_[i] + j + 1]
                    dist.append(lines[lines_[i] + j + 1][:-1])
                    j = j + 1
            mydict[title] = dist
        elif title == 'names of parameters':
            names = []
            j = 0
            while j >= 0:
                testline = lines[lines_[i] + j + 1].strip()
                if not testline:
                    break
                else:
                    x = lines[lines_[i] + j + 1]
                    names.append(lines[lines_[i] + j + 1][:-1])
                    j = j + 1
            mydict[title] = names
        elif title == 'distribution parameters':
            params = []
            j = 0
            while j >= 0:
                testline = lines[lines_[i] + j + 1].strip()
                if not testline:
                    break
                else:
                    x = lines[lines_[i] + j + 1]
                    params.append(np.float32(x.split(" ")))
                    j = j + 1
            mydict[title] = params
        elif title == 'number of samples':
            mydict[title] = int(lines[lines_[i] + 1][:-1])
        elif title == 'number of parameters':
            mydict[title] = int(lines[lines_[i] + 1][:-1])
        ################################################################################################################
        # Latin Hypercube parameters
        elif title == 'criterion':
            mydict[title] = lines[lines_[i] + 1][:-1]
        elif title == 'distance':
            mydict[title] = lines[lines_[i] + 1][:-1]
        elif title == 'iterations':
            mydict[title] = lines[lines_[i] + 1][:-1]
        ################################################################################################################
        #  partially stratified sampling
        elif title == 'design':
            pss_design = []
            j = 0
            while j >= 0:
                testline = lines[lines_[i] + j + 1].strip()
                if not testline:
                    break
                else:
                    x = lines[lines_[i] + j + 1]
                    pss_design.append(int(lines[lines_[i] + j + 1][:-1]))
                    j = j + 1
            mydict[title] = pss_design
        elif title == 'strata':
            pss_strata = []
            j = 0
            while j >= 0:
                testline = lines[lines_[i] + j + 1].strip()
                if not testline:
                    break
                else:
                    x = lines[lines_[i] + j + 1]
                    pss_strata.append(int(lines[lines_[i] + j + 1][:-1]))
                    j = j + 1
            mydict[title] = pss_strata
        ################################################################################################################
        #  stratified sampling
        elif title == 'design':
            sts_design = []
            j = 0
            while j >= 0:
                testline = lines[lines_[i] + j + 1].strip()
                if not testline:
                    break
                else:
                    x = lines[lines_[i] + j + 1]
                    sts_design.append(int(lines[lines_[i] + j + 1][:-1]))
                    j = j + 1
            mydict[title] = sts_design
            print(sts_design)
        ################################################################################################################
        # Markov Chain Monte Carlo simulation
        elif title == 'algorithm':
            mydict[title] = lines[lines_[i] + 1][:-1]
        elif title == 'proposal distribution type':
            dist = []
            j = 0
            while j >= 0:
                testline = lines[lines_[i] + j + 1].strip()
                if not testline:
                    break
                else:
                    dist.append(lines[lines_[i] + j + 1][:-1])
                    j = j + 1
            mydict[title] = dist
        elif title == 'proposal distribution scale':
            dist = []
            j = 0
            while j >= 0:
                testline = lines[lines_[i] + j + 1].strip()
                if not testline:
                    break
                else:
                    dist.append(int(lines[lines_[i] + j + 1][:-1]))
                    j = j + 1
            mydict[title] = dist
        elif title == 'target distribution type':
            dist = []
            j = 0
            while j >= 0:
                testline = lines[lines_[i] + j + 1].strip()
                if not testline:
                    break
                else:
                    dist.append(lines[lines_[i] + j + 1][:-1])
                    j = j + 1
            mydict[title] = dist
        elif title == 'skip':
            mydict[title] = int(lines[lines_[i] + 1][:-1])
        elif title == 'target distribution parameters':
            target_params = list()
            j = 0
            while j >= 0:
                testline = lines[lines_[i] + j + 1].strip()
                if not testline:
                    break
                else:
                    x = lines[lines_[i] + j + 1]
                    target_params.append(np.float32(x.split(" ")))
                    j = j + 1
            mydict[title] = target_params
        elif title == 'seed':
            seed = []
            j = 0
            while j >= 0:
                testline = lines[lines_[i] + j + 1].strip()
                if not testline:
                    break
                else:
                    x = lines[lines_[i] + j + 1]
                    seed.append(np.float32(x.split(" ")))
                    j = j + 1
            mydict[title] = seed
        ################################################################################################################
        # Subset Simulation
        elif title == 'Number of Samples per subset':
            mydict[title] = int(lines[lines_[i] + 1][:-1])
        elif title == 'Conditional probability':
            mydict[title] = np.float32(lines[lines_[i] + 1][:-1])
        elif title == 'Limit-state':
            ls = []
            j = 0
            while j >= 0:
                testline = lines[lines_[i] + j + 1].strip()
                if not testline:
                    break
                else:
                    ls.append(np.float32(lines[lines_[i] + j + 1][:-1]))
                    j = j + 1
                mydict[title] = ls
        elif title == 'Failure probability':
            pf = []
            j = 0
            while j >= 0:
                testline = lines[lines_[i] + j + 1].strip()
                if not testline:
                    break
                else:
                    pf.append(np.float32(lines[lines_[i] + j + 1][:-1]))
                    j = j + 1
                mydict[title] = pf
        ################################################################################################################
        # Stochastic Reduced Order Model
        elif title == 'SROM':
            mydict[title] = lines[lines_[i] + 1][:-1]
        elif title == 'moments':

            seed = []
            j = 0
            while j >= 0:
                testline = lines[lines_[i] + j + 1].strip()
                if not testline:
                    break
                else:
                    x = lines[lines_[i] + j + 1][:-1]
                    seed.append(np.float32(x.split(" ")))
                    j = j + 1
            mydict[title] = seed
        elif title == 'error function weights':
            seed = []
            j = 0
            while j >= 0:
                testline = lines[lines_[i] + j + 1].strip()
                if not testline:
                    break
                else:
                    x = lines[lines_[i] + j + 1][:-1]
                    seed.append(np.float32(x.split(" ")))
                    j = j + 1
            mydict[title] = seed
        elif title == 'properties to match':
            seed = []
            j = 0
            while j >= 0:
                testline = lines[lines_[i] + j + 1].strip()
                if not testline:
                    break
                else:
                    x = lines[lines_[i] + j + 1][:-1]
                    seed.append(np.float32(x.split(" ")))
                    j = j + 1
            mydict[title] = seed
        elif title == 'correlation':
            seed = []
            j = 0
            while j >= 0:
                testline = lines[lines_[i] + j + 1].strip()
                if not testline:
                    break
                else:
                    x = lines[lines_[i] + j + 1][:-1]
                    seed.append(np.float32(x.split(" ")))
                    j = j + 1
            mydict[title] = seed
        elif title == 'weights for distribution':
            seed = []
            j = 0
            while j >= 0:
                testline = lines[lines_[i] + j + 1].strip()
                if not testline:
                    break
                else:
                    x = lines[lines_[i] + j + 1][:-1]
                    seed.append(np.float32(x.split(" ")))
                    j = j + 1
            mydict[title] = seed
        elif title == 'weights for moments':
            seed = []
            j = 0
            while j >= 0:
                testline = lines[lines_[i] + j + 1].strip()
                if not testline:
                    break
                else:
                    x = lines[lines_[i] + j + 1][:-1]
                    seed.append(np.float32(x.split(" ")))
                    j = j + 1
            mydict[title] = seed
        elif title == 'weights for correlation':
            seed = []
            j = 0
            while j >= 0:
                testline = lines[lines_[i] + j + 1].strip()
                if not testline:
                    break
                else:
                    x = lines[lines_[i] + j + 1][:-1]
                    seed.append(np.float32(x.split(" ")))
                    j = j + 1
            mydict[title] = seed
        ################################################################################################################
        # ADD ANY NEW METHOD HERE

        ################################################################################################################
        # ADD ANY NEW METHOD HERE

    return mydict


def save_csv(headers, param_values):

    index = np.array(range(1, param_values.shape[0] + 1)).astype(int)
    param_values = np.hstack((index.reshape(index.shape[0], 1), param_values))
    expand_header = list()
    expand_header.append('Run')
    for i in range(len(headers)):
        expand_header.append(headers[i])
    import csv
    with open('UQpy_Samples.csv', "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerow(expand_header)
        for val in param_values:
            writer.writerow(val)


def save_txt(headers, param_values):
    index = np.array(range(1, param_values.shape[0] + 1)).astype(int)
    param_values = np.hstack((index.reshape(index.shape[0], 1), param_values))
    expand_header = list()
    expand_header.append('Run')
    for i in range(len(headers)):
        expand_header.append(headers[i])

    header = ', '.join(expand_header)
    np.savetxt('UQpy_Samples.txt', param_values, header=str(header), fmt='%0.5f')


def init_sm(data):
    """
    This function is used in order to initialize the selected sampling method and run UQpy from command line.
    :param data: A dictionary containing all data required to run UQpy. The information is enclosed in
                the UQpy_params.txt and the data structure is is created using the ReadInputFile.py
    :type data: dict
    :return:
    """
    ################################################################################################################
    # Add available UQpy methods Here
    valid_sampling_methods = ['mcs', 'lhs', 'mcmc', 'pss', 'sts']
    valid_reliability_methods = ['SuS']

    ################################################################################################################
    # Check if requested method is available

    if 'method' in data:
        if data['method'] not in valid_sampling_methods or data['method'] not in valid_reliability_methods:
            raise NotImplementedError("method - %s not available in UQpy" % data['method'])
    else:
        raise NotImplementedError("No UQpy method was requested")

    ################################################################################################################
    # Monte Carlo simulation block.
    # Mandatory properties(4): 1. Number of parameters, 2. distribution, 3. distribution parameters 4. Number of samples
    # Optional properties(0):

    if data['method'] == 'mcs':

        # Mandatory
        if 'number of samples' not in data:
            data['number of samples'] = None
        if 'distribution type' not in data:
            raise NotImplementedError("Exit UQpy: Distribution of random variables not defined.")
        if 'distribution parameters' not in data:
            raise NotImplementedError("Distribution parameters not provided. ")
        if 'number of parameters' not in data:
            data['number of parameters'] = None

    ################################################################################################################
    # Latin Hypercube simulation block.
    # Mandatory properties(4): 1. Number of parameters, 2. distribution, 3. distribution parameters 4. Number of samples
    # Optional properties(3):  1. Criterion, 2. Metric, 3. Iterations

    if data['method'] == 'lhs':
        # Mandatory
        if 'number of parameters' not in data:
            data['number of parameters'] = None
        if 'number of samples' not in data:
            data['number of samples'] = None
        if 'distribution type' not in data:
            raise NotImplementedError("Exit UQpy: Distribution of random variables not defined.")
        if 'distribution parameters' not in data:
            raise NotImplementedError("Exit UQpy: Distribution parameters not defined.")

        # Optional
        if 'criterion' not in data:
            data['criterion'] = None
        if 'distance' not in data:
            data['distance'] = None
        if 'iterations' not in data:
            data['iterations'] = None

    ####################################################################################################################
    # Markov Chain Monte Carlo simulation block.
    # Mandatory properties(4):  1. target distribution, 2. target distribution parameters, 3. Number of samples,
    #                           4. Number of parameters
    #  Optional properties(5): 1. Proposal distribution, 2. proposal width, 3. Seed, 4. skip samples (avoid burn-in),
    #                          5. algorithm

    if data['method'] == 'mcmc':
        # Mandatory
        if 'number of parameters' not in data:
            raise NotImplementedError('Exit UQpy: Number of random variables not defined.')
        if 'target distribution type' not in data:
            raise NotImplementedError("Exit UQpy: Target distribution type not defined.")
        if 'target distribution parameters' not in data:
            raise NotImplementedError("Exit UQpy: Target distribution parameters not defined.")
        if 'number of samples' not in data:
            raise NotImplementedError('Exit UQpy: Number of samples not defined.')
        # Optional
        if 'seed' not in data:
            data['seed'] = None
        if 'skip' not in data:
            data['skip'] = None
        if 'proposal distribution type' not in data:
            data['proposal distribution type'] = None
        if 'proposal distribution width' not in data:
            data['proposal distribution width'] = None
        if 'algorithm' not in data:
            data['algorithm'] = None

    ################################################################################################################
    # Partially stratified sampling  block.
    # Mandatory properties (4):  1. distribution, 2. distribution parameters, 3. design, 4. strata
    # Optional properties(1): 1. Number of parameters

    if data['method'] == 'pss':

        # Mandatory
        if 'distribution type' not in data:
            raise NotImplementedError("Exit UQpy: Distribution of random variables not defined.")
        elif 'distribution parameters' not in data:
            raise NotImplementedError("Exit UQpy: distribution parameters not defined.")
        if 'design' not in data:
            raise NotImplementedError("Exit UQpy: pss design not defined.")
        if 'strata' not in data:
            raise NotImplementedError("Exit UQpy: pss strata not defined.")

        # Optional
        if 'number of parameters' not in data:
            data['number of parameters'] = None

    ################################################################################################################
    # Stratified sampling block.
    # Mandatory properties(3):  1. distribution, 2. distribution parameters, 3. design
    # Optional properties(1): 1. Number of parameters

    if data['method'] == 'sts':
        # Mandatory
        if 'distribution type' not in data:
            raise NotImplementedError("Exit UQpy: Distribution of random variables not defined.")
        elif 'distribution parameters' not in data:
            raise NotImplementedError("Exit UQpy: distribution parameters not defined.")
        if 'design' not in data:
            raise NotImplementedError("Exit UQpy: sts design not defined.")

        # Optional
        if 'number of parameters' not in data:
            data['number of parameters'] = None


    ####################################################################################################################
    # Subset Simulation simulation block.
    # Necessary MCMC parameters:  1. Proposal pdf, 2. Proposal width, 3. Target pdf, 4. Target pdf parameters
    #                             5. algorithm
    # Optional: 1. Seed, 2. skip

    if data['Method'] == 'SuS':
        if 'Proposal distribution' not in data:
            data['Proposal distribution'] = None

        if 'Target distribution' not in data:
            data['Target distribution'] = None

        if 'Target distribution parameters' not in data:
            data['Target distribution parameters'] = None

        if 'Proposal distribution scale' not in data:
            data['Proposal distribution scale'] = None

        if 'MCMC algorithm' not in data:
            data['MCMC algorithm'] = None

        if 'Number of Samples per subset' not in data:
            data['Number of Samples per subset'] = None

        if 'Conditional probability' not in data:
            data['Conditional probability'] = None

        if 'Limit-state' not in data:
            data['Limit-state'] = None

    ####################################################################################################################
    # Check any NEW  SAMPLING METHOD HERE
    #
    #


########################################################################################################################
########################################################################################################################
#                                       RUN THE SELECTED METHOD                                                        #


def run_sm(data):
    """
    Run the selected UQpy sampling method.
    :param data:
    :return: An array containing the samples as well as the attributes of the proposed method.
    """
    ################################################################################################################
    # Run Monte Carlo simulation
    if data['method'] == 'mcs':
        print("\nRunning  %k \n", data['method'])
        rvs = MCS(dimension=data['number of parameters'], pdf_type=data['distribution type'],
                  pdf_params=data['distribution parameters'],
                  nsamples=data['number of samples'])

    ################################################################################################################
    # Run Latin Hypercube sampling
    elif data['method'] == 'lhs':
        print("\nRunning  %k \n", data['method'])
        rvs = LHS(dimension=data['number of parameters'], pdf_type=data['distribution type'],
                  pdf_params=data['distribution parameters'],
                  nsamples=data['number of samples'], lhs_metric=data['distance'],
                  lhs_iter=data['iterations'], lhs_criterion=data['criterion'])

    ################################################################################################################
    # Run partially stratified sampling
    elif data['method'] == 'pss':
        print("\nRunning  %k \n", data['method'])
        rvs = PSS(dimension=data['number of parameters'], pdf_type=data['distribution type'],
                  pdf_params=data['distribution parameters'],
                  pss_design=data['design'], pss_strata=data['strata'])

    ################################################################################################################
    # Run STS sampling

    elif data['method'] == 'sts':
        print("\nRunning  %k \n", data['method'])
        rvs = STS(dimension=data['number of parameters'], pdf_type=data['distribution type'],
                  pdf_params=data['distribution parameters'], sts_design=data['design'])

    ################################################################################################################
    # Run Markov Chain Monte Carlo sampling

    elif data['method'] == 'mcmc':
        print("\nRunning  %k \n", data['method'])
        rvs = MCMC(dimension=data['number of parameters'], pdf_target_type=data['target distribution type'],
                   algorithm=data['algorithm'], pdf_proposal_type=data['proposal distribution type'],
                   pdf_proposal_scale=data['proposal distribution width'],
                   pdf_target_params=data['target distribution parameters'], seed=data['seed'],
                   nburn=data['skip'], nsamples=data['number of samples'])

    ##################################################################################################################
    # Run Subset Simulation
    if data['method'] == 'SuS':
        print("\nRunning  %k \n", data['Method'])

    return rvs


########################################################################################################################

def RunCommandLine(argparseobj):
    os.system('clear')
    args = argparseobj

    ################################################################################################################
    # Read  UQpy parameter file

    current_dir = os.getcwd()
    os.chdir(args.Model_directory)

    if not os.path.isfile('UQpy_Params.txt'):
        print("Error: UQpy parameters file does not exist")
        sys.exit()
    else:
        data = readfile('UQpy_Params.txt')

    ################################################################################################################
    # Run UQpy

    print("\nExecuting UQpy...\n")

    # Run Selected method

    rvs = run_sm(data)

    np.savetxt('UQpy_Samples.txt', rvs.samples, fmt='%0.5f')

    # Save the samples in a .csv file
    if 'names of parameters' not in data:
        import itertools
        data['names of parameters'] = list(itertools.repeat('#name', rvs.samples.shape[1]))

    save_csv(data['names of parameters'], rvs.samples)

    ################################################################################################################

    ################################################################################################################
    # If a model is provided then run it
    if args.model_script is not None:
        print(args.model_script)
        RunModel(cpu=args.CPU, model_type=args.model_type, model_script=args.model_script,
                 input_script=args.Input_Shell_Script, output_script=args.Output_Shell_Script,
                 dimension=data['number of parameters'])

    ################################################################################################################
    print("\nSuccessful execution of UQpy\n\n")


UQpy_commandLine = \
UQpy_commandLine_Usage = """python UQpy.py --{options}"""

if __name__ == '__main__':

    UQpymainDir = os.path.realpath(__file__)
    UQpydirectory = os.path.dirname(UQpymainDir)

    parser = ArgumentParser(description=UQpy_commandLine,
                            usage=UQpy_commandLine_Usage)

    parser.add_argument("--dir", dest="Model_directory", action="store",
                        default=None, help="Specify the location of the model's directory.")

    parser.add_argument("--model_type", default=None,
                        dest="model_type", action="store",
                        help="Specify the type of the model: python, abaqus, opensees, MATLAB")

    parser.add_argument("--model_script", default=None,
                        dest="model_script", action="store",
                        help="Specify the name of the script  (*.sh, *.py) that runs the model")

    parser.add_argument("--input_script", default=None,
                        dest="Input_Shell_Script", action="store",
                        help="Specify the name of the shell script  *.sh used to transform  the output of UQpy"
                             " (UQpyOut_*.txt file) into the appropriate model input file ")

    parser.add_argument("--output_script", default=None,
                        dest="Output_Shell_Script", action="store",
                        help="Specify the name of the shell script  *.sh used to transform  the output of the model"
                             " into the appropriate UQpy input file (UQpyInp_*.txt) ")

    parser.add_argument("--cpu", dest="CPU", action="store",
                        default=0, type=int, help="Number of local cpu to be used for the analysis")

    args, unknown = parser.parse_known_args()

    if len(unknown) > 0:
        print("Warning:  The following unrecognized arguments were ignored:")
        print(unknown)

    if len(sys.argv) > 1:
        if args.Model_directory is None:
            print("Error: A model directory needs to be specified")
            sys.exit()

        if args.model_script is not None:
            if args.model_type != 'python':
                if args.Input_Shell_Script is None:
                    print("Error: Shell scripts for communication between UQpy and the model are required. Type --help"
                          "for more information")
                    sys.exit()

                elif args.Output_Shell_Script is None:
                    print("Error: Shell scripts for communication between UQpy and the model are required. Type --help"
                          "for more information")
                    sys.exit()

        RunCommandLine(args)