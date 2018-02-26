import numpy as np
import os
import sys
import shutil
from UQpyLibraries.SampleMethods import transform_pdf
import warnings


class RunCommandLine:

    def __init__(self, argparseobj):

        self.args = argparseobj
        ################################################################################################################
        # Create a unique temporary directory. Remove after completion.
        self.current_dir = os.getcwd()
        dir_path = os.path.join(self.current_dir, 'tmp')
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs('tmp', exist_ok=True)
        self.args.WorkingDir = os.path.join(os.sep, self.current_dir, 'tmp')

        src_files = os.listdir(self.args.Model_directory)
        for file_name in src_files:
            full_file_name = os.path.join(self.args.Model_directory, file_name)

            # Copy ALL files from Model_directory to Working_directory
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, self.args.WorkingDir)

        ################################################################################################################
        # Run UQpy

        print("\nExecuting UQpy...\n")
        self.run_uq()

    def run_uq(self):

        os.chdir(os.path.join(self.current_dir, self.args.WorkingDir))

        ################################################################################################################
        # Read  UQpy parameter file
        from UQpyLibraries import ReadInputFile

        if not os.path.isfile('UQpy_Params.txt'):
            print("Error: UQpy parameters file does not exist")
            sys.exit()
        else:
            data = ReadInputFile.readfile('UQpy_Params.txt')

        ################################################################################################################
        # Initialize the requested UQpy method: Check if all necessary parameters are defined in the UQpyParams.txt file
        init_sm(data)

        ################################################################################################################
        # Run the requested UQpy method and save the samples into file 'UQpyOut.txt'
        samples_01 = run_sm(data)

        # Transform samples to the original parameter space
        if data['Method'] != 'mcmc':
            samples = transform_pdf(samples_01, data['Probability distribution (pdf)'],
                                    data['Probability distribution parameters'])
        else:
            samples = samples_01

        if data['SROM'] is True:
            from UQpyLibraries.SampleMethods import SROM
            samples = SROM(samples=samples, nsamples=data['Number of Samples'],
                       marginal=data['Probability distribution (pdf)'], moments=data['Moments'],
                       weights_errors=data['Error function weights'], weights_function=data['Sample weights'],
                   properties=data['Properties to match'])


        # Save the samples in a .txt file
        save_txt(data['Names of random variables'], samples)

        # Save the samples in a .csv file
        save_csv(data['Names of random variables'], samples)


        ################################################################################################################
        # Split the samples into chunks in order to sent to each processor in case of parallel computing, else save them
        # into file UQpyOut.txt
        if self.args.CPUs != 0:
            if samples.shape[0] <= self.args.CPUs:
                self.args.CPUs = samples.shape[0]
                self.args.CPUs_reduced = True
                print('The number of CPUs used is\n %', samples.shape[0])
            else:
                self.args.CPUs_reduced = False
            chunk_samples_cores(data, samples, self.args)
        else:
            self.args.CPUs_reduced = False

        ################################################################################################################
        # If a model is provided then run it, else move the data to directory simUQpyOut/ , delete the temp/ directory
        # and terminate the program

        if self.args.Solver is None:
            full_file_name = os.path.join(self.args.WorkingDir, 'UQpyOut.txt')
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, self.args.Output_directory)

        else:
            RunModel(model_script=self.args.Solver, input_script=self.args.Input_Shell_Script,
                     output_script=self.args.Output_Shell_Script,
                     cpu=self.args.CPUs, cpu_red=self.args.CPUs_reduced)

            src_files = [filename for filename in os.listdir(self.args.WorkingDir) if filename.startswith("UQpyInput_")]
            _files = []
            for file in src_files:
                file_new = file.replace("UQpyInput_", "Model_")
                os.rename(file, file_new)
                _files.append(file_new)
            _files.append('UQpyOut.csv')
            _files.append('UQpyOut.txt')

            for file_name in _files:
                full_file_name = os.path.join(self.args.WorkingDir, file_name)
                shutil.copy(full_file_name, self.args.Output_directory)

        shutil.rmtree(self.args.WorkingDir)
        shutil.move(self.args.Output_directory, self.args.Model_directory)

        print("\nSuccessful execution of UQpy\n\n")


class RunModel:

    """
    A class used to run the computational model.

    :param cpu:  Number of CPUs for parallel processing
    :type cpu: int
    :param model_script: UQpy_Model.sh script
    :param input_script: UQpy_Input.sh script
    :param output_script: UQpy_Output.sh script
    :param cpu_red: Number of cores to be used

    """
    def __init__(self,  cpu=None, model_script=None, input_script=None, output_script=None, cpu_red=None):

        self.CPUs = cpu
        self.model_script = model_script
        self.input_script = input_script
        self.output_script = output_script
        self.current_dir = os.getcwd()
        self.CPUs_reduced = cpu_red

        if self.CPUs != 0:
            import multiprocessing
            n_cpu = multiprocessing.cpu_count()
            if self.CPUs > n_cpu:
                print("Error: You have available {0:1d} CPUs. Start parallel computing  using {0:1d} CPUs".format(np))
                self.CPUs = n_cpu
            self.ParallelProcessing = True
            self.values = self.multi_core()

        else:
            self.ParallelProcessing = False
            self.values = self.run_model()

    def run_model(self):
        import time
        start_time = time.time()

        # Load the UQpyOut.txt
        values = np.loadtxt('UQpyOut.txt', dtype=np.float32)

        print("\nEvaluating the model...\n")
        model_eval = []
        for i in range(values.shape[0]):
            # Write each value of UQpyOut.txt into a *.txt file
            with open('TEMP_val_{0}.txt'.format(i), 'wb') as f:
                np.savetxt(f, values[i, :], fmt='%0.5f')

            # Run the Input_Shell_Script.sh in order to create the input file for the model
            join_input_script = './{0} {1}'.format(self.input_script, i)
            os.system(join_input_script)

            # Run the Model.sh in order to run the model
            join_model_script = './{0} {1}'.format(self.model_script, i)
            os.system(join_model_script)

            # Run the Output_Shell_Script.sh  in order to create the input file of the model for UQpy
            join_output_script = './{0} {1}'.format(self.output_script, i)
            os.system(join_output_script)

            model_eval.append(np.loadtxt('UQpyInput_{}.txt'.format(i), dtype=np.float32))

        end_time = time.time()
        print(end_time - start_time, "(sec)- Serial")
        return model_eval

    def run_parallel_model(self, args, multi=False, queue=0):
        import os
        from multiprocessing import Lock
        j = args

        # Define the executable shell scripts for the model

        # Load the UQpyOut.txt
        values = np.loadtxt('LocalChunk_{0}.txt'.format(j+1), dtype=np.float32)
        index = np.loadtxt('LocalChunk_index_{0}.txt'.format(j+1))

        model_eval = []
        if self.CPUs_reduced is True or index.size == 1:

            # Write each value of UQpyOut.txt into a *.txt file
            np.savetxt('TEMP_val_{0}.txt'.format(int(index)), values, newline=' ', delimiter=',',  fmt='%0.5f')

            # Run the Input_Shell_Script.sh in order to create the input file for the model
            join_input_script = './{0} {1}'.format(self.input_script, int(index))
            os.system(join_input_script)

            # Run the Model.sh in order to run the model
            join_model_script = './{0} {1}'.format(self.model_script, int(index))
            os.system(join_model_script)

            # Run the Output_Shell_Script.sh  in order to create the input file of the model for UQpy
            join_output_script = './{0} {1}'.format(self.output_script, int(index))
            os.system(join_output_script)

            model_eval.append(np.loadtxt('UQpyInput_{0}.txt'.format(int(index)), dtype=np.float32))

            if multi:
                queue.put(model_eval)

        else:

            count = 0
            for i in index:
                lock = Lock()

                lock.acquire()  # will block if lock is already held

                # Write each value of UQpyOut.txt into a *.txt file
                np.savetxt('TEMP_val_{0}.txt'.format(int(i)), values[count, :], newline=' ', delimiter=',', fmt='%0.5f')

                # Run the Input_Shell_Script.sh in order to create the input file for the model
                join_input_script = './{0} {1}'.format(self.input_script, int(i))
                os.system(join_input_script)

                # Run the Model.sh in order to run the model
                join_model_script = './{0} {1}'.format(self.model_script, int(i))
                os.system(join_model_script)

                # Run the Output_Shell_Script.sh  in order to create the input file of the model for UQpy
                join_output_script = './{0} {1}'.format(self.output_script, int(i))
                os.system(join_output_script)

                model_eval.append(np.loadtxt('UQpyInput_{0}.txt'.format(int(i)), dtype=np.float32))
                count = count + 1
                lock.release()

            if multi:
                queue.put(model_eval)

        return model_eval

    def multi_core(self):
        from multiprocessing import Process
        from multiprocessing import Queue
        import time

        start_time = time.time()

        print("\nEvaluating the model...\n")

        results = []
        queues = [Queue() for i in range(self.CPUs)]

        args = [(i, True, queues[i]) for i in range(self.CPUs)]

        jobs = [Process(target=self.run_parallel_model, args=a) for a in args]

        for j in jobs:
            j.start()
        for q in queues:
            results.append(q.get())
        for j in jobs:
            j.join()
        end_time = time.time()
        print(end_time - start_time, "(sec)-Parallel")
        return results


def chunk_samples_cores(data, samples, args):

    header = ', '.join(data['Names of random variables'])
    # In case of parallel computing divide the samples into chunks in order to sent to each processor
    chunks = args.CPUs
    if args.CPUs_reduced is True:
        for i in range(args.CPUs):
            np.savetxt('LocalChunk_{0}.txt'.format(i+1), samples[range(i-1, i), :], header=str(header), fmt='%0.5f')
            np.savetxt('LocalChunk_index_{0}.txt'.format(i+1), np.array(i).reshape(1,))

    else:
        size = np.array([np.ceil(samples.shape[0]/chunks) for i in range(args.CPUs)]).astype(int)
        dif = np.sum(size) - samples.shape[0]
        count = 0
        for k in range(dif):
            size[count] = size[count] - 1
            count = count + 1
        for i in range(args.CPUs):
            if i == 0:
                lines = range(size[i])
            else:
                lines = range(int(np.sum(size[:i])), int(np.sum(size[:i+1])))
            np.savetxt('LocalChunk_{0}.txt'.format(i+1), samples[lines, :], header=str(header), fmt='%0.5f')
            np.savetxt('LocalChunk_index_{0}.txt'.format(i+1), lines)


def chunk_samples_nodes(data, samples, args):

    header = ', '.join(data['Names of random variables'])

    # In case of cluster divide the samples into chunks in order to sent to each processor
    chunks = args.nodes
    size = np.array([np.ceil(samples.shape[0]/chunks) in range(args.nodes)]).astype(int)
    dif = np.sum(size) - samples.shape[0]
    count = 0
    for k in range(dif):
        size[count] = size[count] - 1
        count = count + 1
    for i in range(args.nodes):
        if i == 0:
            lines = range(0, size[i])
        else:
            lines = range(int(np.sum(size[:i])), int(np.sum(size[:i+1])))

        np.savetxt('ClusterChunk_{0}.txt'.format(i+1), samples[lines, :], header=str(header), fmt='%0.5f')
        np.savetxt('ClusterChunk_index_{0}.txt'.format(i+1), lines)


def init_sm(data):

    ################################################################################################################
    # Add available methods Here
    valid_methods = ['mcs', 'lhs', 'mcmc', 'pss', 'sts', 'SuS', 'srom']

    ################################################################################################################
    # Check if requested method is available

    if 'Method' in data.keys():
        if data['Method'] not in valid_methods:
            raise NotImplementedError("Method - %s not available" % data['Method'])
    else:
        raise NotImplementedError("No sampling method was provided")

    ################################################################################################################
    # Monte Carlo simulation block.
    # Necessary parameters:  1. Probability distribution, 2. Probability distribution parameters
    # Optional:

    if data['Method'] == 'mcs':
        if 'Number of Samples' not in data.keys():
            data['Number of Samples'] = None
            warnings.warn("Number of samples not provided. Default number is 100")
        if 'Probability distribution (pdf)' not in data.keys():
            raise NotImplementedError("Probability distribution not provided")
        elif 'Probability distribution parameters' not in data.keys():
            raise NotImplementedError("Probability distribution parameters not provided")

    ################################################################################################################
    # Latin Hypercube simulation block.
    # Necessary parameters:  1. Probability distribution, 2. Probability distribution parameters
    # Optional: 1. Criterion, 2. Metric, 3. Iterations

    if data['Method'] == 'lhs':
        if 'Number of Samples' not in data:
            data['Number of Samples'] = None
            warnings.warn("Number of samples not provided. Default number is 100")
        if 'Probability distribution (pdf)' not in data:
            raise NotImplementedError("Probability distribution not provided")
        if 'Probability distribution parameters' not in data:
            raise NotImplementedError("Probability distribution parameters not provided")
        if 'LHS criterion' not in data:
            data['LHS criterion'] = 'random'
            warnings.warn("LHS criterion not defined. The default is centered")
        if 'distance metric' not in data:
            data['distance metric'] = 'euclidean'
            warnings.warn("Distance metric for the LHS not defined. The default is Euclidean")
        if 'iterations' not in data:
            data['iterations'] = 1000
            warnings.warn("Iterations for the LHS not defined. The default number is 1000")

    ####################################################################################################################
    # Markov Chain Monte Carlo simulation block.
    # Necessary parameters:  1. Proposal pdf, 2. Probability pdf width, 3. Target pdf, 4. Target pdf parameters
    #                        5. algorithm
    # Optional: 1. Seed, 2. Burn-in

    if data['Method'] == 'mcmc':
        if 'Number of Samples' not in data:
            data['Number of Samples'] = 100
            warnings.warn("Number of samples not provided. Default number is 100")
        if 'MCMC algorithm' not in data:
            warnings.warn("MCMC algorithm not provided. The Metropolis-Hastings algorithm will be used")
            data['MCMC algorithm'] = 'MH'
        else:
            if data['MCMC algorithm'] not in ['MH', 'MMH']:
                warnings.warn("MCMC algorithm not available. The Metropolis-Hastings algorithm will be used")
                data['MCMC algorithm'] = 'MH'
        if 'Proposal distribution' not in data:
            raise NotImplementedError("Proposal distribution not provided")
        if 'Proposal distribution width' not in data:
            raise NotImplementedError("Proposal distribution parameters (width) not provided")
        if data['MCMC algorithm'] == 'MH':
            if 'Number of random variables' not in data:
                if 'Names of random variables ' not in data:
                    raise NotImplementedError("Dimension of the problem not specified")
                else:
                    data['Number of random variables'] = len(data['Names of random variables'])
            if 'Target distribution parameters' not in data:
                raise NotImplementedError("Target distribution parameters not provided")
        if data['MCMC algorithm'] == 'MMH':
            if 'Marginal Target distribution parameters' not in data:
                raise NotImplementedError("Marginal Target distribution parameters not provided")
        if 'Burn-in samples' not in data:
            data['Burn-in samples'] = 1
            warnings.warn("Number of samples to skip in order to avoid burn-in not provided."
                          "The default will be set equal to 1")
        if 'seed' not in data:
            data['seed'] = np.zeros(len(data['Names of random variables']))
            warnings.warn("Chain will start from 0")

    ################################################################################################################
    # Partially stratified sampling (PSS) block.
    # Necessary parameters:  1. pdf, 2. pdf parameters 3. pss design 3. pss strata
    # Optional:
    # TODO: PSS block
    ################################################################################################################
    # Stratified sampling (STS) block.
    # Necessary parameters:  1. pdf, 2. pdf parameters 3. sts design
    # Optional:
    # TODO: STS block
    ################################################################################################################
    # HERE YOU ADD CHECKS FOR ANY NEW METHOD ADDED
    # Necessary parameters:
    # Optional:
    # TODO: Subset Simulation block
        ################################################################################################################
        # Partially stratified sampling (PSS) block.
        # Necessary parameters:  1. pdf, 2. pdf parameters 3. pss design 3. pss strata
        # Optional:
        # TODO: PSS block
        ################################################################################################################
        # Stratified sampling (STS) block.
        # Necessary parameters:  1. pdf, 2. pdf parameters 3. sts design
        # Optional:
        # TODO: STS block
        ################################################################################################################
        # Stochastic Reduced Order Model (SROM) block.
        # Necessary parameters:  1. marginal pdf, 2. moments 3. Error function weights
        # Optional: 1. Properties to match 2. Error function weights

        if data['Method'] == 'srom':
            if 'Probability distribution (pdf)' not in data:
                raise NotImplementedError("Probability distribution not provided")
            if 'Moments' not in data:
                raise NotImplementedError("Moments not provided")
            if 'Error function weights' not in data:
                raise NotImplementedError("Error function weights not provided")
            if 'Properties to match' not in data:
                data['Properties to match'] = None
                warnings.warn("Properties to match not defined. The default is [1, 1, 0]")
            if 'Error function weights' not in data:
                data['Error function weights'] = None
                warnings.warn("Error function weights not defined. The default is equal weights to each sample")

        ################################################################################################################
        # HERE YOU ADD CHECKS FOR ANY NEW METHOD ADDED
        # Necessary parameters:
        # Optional:
        # TODO: Subset Simulation block


def run_sm(data):
    ################################################################################################################
    # Run Monte Carlo simulation
    if data['Method'] == 'mcs':
        from UQpyLibraries.SampleMethods import MCS
        print("\nRunning  %k \n", data['Method'])
        rvs = MCS(pdf=data['Probability distribution (pdf)'],
                  pdf_params=data['Probability distribution parameters'],
                  nsamples=data['Number of Samples'])

    ################################################################################################################
    # Run Latin Hypercube sampling
    elif data['Method'] == 'lhs':
        from UQpyLibraries.SampleMethods import LHS
        print("\nRunning  %k \n", data['Method'])
        rvs = LHS(pdf=data['Probability distribution (pdf)'],
                  pdf_params=data['Probability distribution parameters'],
                  nsamples=data['Number of Samples'], lhs_metric=data['distance metric'],
                  lhs_iter=data['iterations'], lhs_criterion=data['LHS criterion'])

    ################################################################################################################
    # Run partially stratified sampling
    elif data['Method'] == 'pss':
        from UQpyLibraries.SampleMethods import PSS
        print("\nRunning  %k \n", data['Method'])
        rvs = PSS(pdf=data['Probability distribution (pdf)'],
                  pdf_params=data['Probability distribution parameters'],
                  pss_design=data['PSS design'], pss_strata=data['PSS strata'])

    ################################################################################################################
    # Run Markov Chain Monte Carlo sampling

    elif data['Method'] == 'mcmc':
        from UQpyLibraries.SampleMethods import MCMC
        print("\nRunning  %k \n", data['Method'])
        rvs = MCMC(dim=data['Number of random variables'], pdf_target=data['Target distribution'],
                   mcmc_algorithm=data['MCMC algorithm'], pdf_proposal=data['Proposal distribution'],
                   pdf_proposal_width=data['Proposal distribution width'],
                   pdf_target_params=data['Target distribution parameters'], mcmc_seed=data['seed'],
                   pdf_marg_target_params=data['Marginal Target distribution parameters'],
                   pdf_marg_target=data['Marginal target distribution'],
                   mcmc_burnIn=data['Burn-in samples'], nsamples=data['Number of Samples'])
    ################################################################################################################
    # Run stratified sampling
    # TODO: PSS sampling

    ################################################################################################################
    # Run ANY NEW METHOD HERE
    # TODO: STS sampling

    elif data['Method'] == 'sts':
        from UQpyLibraries.SampleMethods import STS
        print("\nRunning  %k \n", data['Method'])
        rvs = STS(pdf=data['Probability distribution (pdf)'],
                  pdf_params=data['Probability distribution parameters'], sts_design=data['STS design'])

    ################################################################################################################
    # Run ANY NEW METHOD HERE
    # TODO: ROM sampling

    ################################################################################################################
    # Run ANY NEW METHOD HERE
    # TODO: Subset Simulation

    return rvs.samples


def save_csv(headers, param_values):

    index = np.array(range(1, param_values.shape[0] + 1)).astype(int)
    param_values = np.hstack((index.reshape(index.shape[0], 1), param_values))
    HEADER=[]
    HEADER.append('Run')
    for i in range(len(headers)):
        HEADER.append(headers[i])
    import csv
    with open('UQpyOut.csv', "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerow(HEADER)
        for val in param_values:
            writer.writerow(val)


def save_txt(headers, param_values):

    header = ', '.join(headers)
    np.savetxt('UQpyOut.txt', param_values, header=str(header), fmt='%0.5f')