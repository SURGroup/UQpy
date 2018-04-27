import os
import shutil
import UQpy as uq
import numpy as np
import sys


class RunCommandLine:

    def __init__(self, argparseobj):
        os.system('clear')
        self.args = argparseobj

        ################################################################################################################
        # Read  UQpy parameter file

        os.chdir(os.path.join(os.getcwd(), self.args.Model_directory))

        from UQpy  import ReadInputFile
        if not os.path.isfile('UQpy_Params.txt'):
            print("Error: UQpy parameters file does not exist")
            sys.exit()
        else:
            from UQpy.ReadInputFile import readfile
            data = readfile('UQpy_Params.txt')

        ################################################################################################################
        # Run UQpy

        print("\nExecuting UQpy...\n")
        ################################################################################################################

        # Run Selected method
        if data['method'] in ['SuS']:
            self.run_reliability(data)
        elif data['method'] in ['mcs', 'lhs', 'mcmc', 'pss', 'sts']:
            self.run_uq(data)

    def run_uq(self, data):

        # Steps:
        # Initialize the sampling method (Check if UQpy_Params.txt contains all the necessary information)
        # Actually run the selected sampling method in U(0, 1) and transform to the original space
        # Save the samples in a .txt (or .csv) file
        # If a solver (black box model) is provided then:
        #                              If parallel processing is selected: Split the samples into chunks
        # Run the model
        # Save the model evaluations

        # Brute-force sampling methods. Set the adaptivity flag False
        self.args.Adaptive = False
        ################################################################################################################
        # Initialize the requested UQpy method: Check if all necessary parameters are defined in the UQpyParams.txt file
        from UQpy.SampleMethods import init_sm, run_sm
        init_sm(data)

        ################################################################################################################
        # Run the requested UQpy method
        rvs = run_sm(data)

        # Save the samples in a .txt file
        np.savetxt('UQpy_Samples.txt', rvs.samples, fmt='%0.5f')

        # Save the samples in a .csv file
        if 'names of parameters' not in data:
            import itertools
            data['names of parameters'] = list(itertools.repeat('#name', rvs.samples.shape[1]))

        save_csv(data['names of parameters'], rvs.samples)

        ################################################################################################################
        # If a model is provided then run it

        if self.args.Solver is not None:
            RunModel(self.args.CPUs, self.args.Solver, self.args.Input_Shell_Script, self.args.Output_Shell_Script,
                     self.args.Adaptive, rvs.dimension)


        ################################################################################################################
        print("\nSuccessful execution of UQpy\n\n")

    def run_reliability(self, data):
        from UQpy.Reliability import init_rm, run_rm
        init_rm(data)
        if data['method'] == 'SuS':
            from UQpy.Reliability import SubsetSimulation
            self.args.CPUs_flag = True
            self.args.ParallelProcessing = False
            self.args.Adaptive = True
            sus = run_rm(self, data)

            # Save the samples in a .txt file
            np.savetxt('UQpy_Samples.txt', sus.samples, fmt='%0.5f')

            # Save the samples in a .csv file
            save_csv(data['Names of random variables'], sus.samples)

            # Save the probability of failure in a .txt file
            print(sus.pf)
            with open('PF.txt', 'wb') as f:
                np.savetxt(f, [sus.pf], fmt='%0.6f')
                np.savetxt(f, [sus.cov], fmt='%0.6f')
        ################################################################################################################
        # Move the data to directory simUQpyOut/ , delete the temp/ directory
        # and terminate the program

        _files = list()
        _files.append('UQpy_Samples.csv')
        _files.append('UQpy_Samples.txt')
        _files.append('PF.txt')

        for file_name in _files:
            full_file_name = os.path.join(self.args.WorkingDir, file_name)
            shutil.copy(full_file_name, self.args.Output_directory)

        shutil.rmtree(self.args.WorkingDir)
        shutil.move(self.args.Output_directory, self.args.Model_directory)

        ################################################################################################################
        print("\nSuccessful execution of UQpy\n\n")


class RunModel:

    """
    A class used to run the computational model.

    :param cpu:
    :param solver:
    :param input_:
    :param output_:
    :param adaptive:
    :param dimension:

    """
    def __init__(self, cpu=None, solver=None, input_=None, output_=None, adaptive=None, dimension=None):

        self.CPUs = cpu
        self.model_script = solver
        self.input_script = input_
        self.output_script = output_
        self.Adaptive = adaptive
        self.dimension = dimension

        import shutil
        current_dir = os.getcwd()
        ################################################################################################################
        # Create a unique temporary directory. Remove after completion.
        folder_name = 'simUQpyOut'
        output_directory = os.path.join(os.sep, current_dir, folder_name)

        model_files = list()
        for fname in os.listdir(current_dir):
            path = os.path.join(current_dir, fname)
            if not os.path.isdir(path):
                model_files.append(path)

        dir_path = os.path.join(current_dir, 'tmp')
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs('tmp', exist_ok=False)
        work_dir = os.path.join(os.sep, current_dir, 'tmp')

        # copy UQ_samples.txt to working-directory
        for file_name in model_files:
            full_file_name = os.path.join(current_dir, file_name)
            shutil.copy(full_file_name, work_dir)

        os.chdir(os.path.join(current_dir, work_dir))

        if self.CPUs != 0 and self.CPUs is not None:
            parallel_processing = True
            import multiprocessing
            n_cpu = multiprocessing.cpu_count()
            if self.CPUs > n_cpu:
                print("Error: You have available {0:1d} CPUs. Start parallel computing using {0:1d} CPUs".format(n_cpu))
                self.CPUs = n_cpu
        else:
            parallel_processing = False

        print("\nEvaluating the model...\n")
        import time
        start_time = time.time()
        if parallel_processing is True:
            self.values = self.multi_core()
        else:
            self.values = self.run_model()
        end_time = time.time()
        print('Total time:', end_time - start_time, "(sec)")


        ################################################################################################################
        # Move the data to directory simUQpyOut

        os.makedirs(output_directory, exist_ok=True)

        path = os.path.join(current_dir, work_dir)

        src_files = [filename for filename in os.listdir(path) if filename.startswith("Model_")]

        for file_name in src_files:
            full_file_name = os.path.join(path, file_name)
            shutil.copy(full_file_name, output_directory)


        ################################################################################################################
        # Delete the tmp working directory directory
        shutil.rmtree(work_dir)
        os.chdir(current_dir)

    def run_model(self):

        # Load the UQpyOut.txt
        values = np.loadtxt('UQpy_Samples.txt', dtype=np.float32)
        if self.Adaptive is True:
            values = values.reshape(1, values.shape[0])

        if self.dimension == 1:
            values = values.reshape(values.shape[0], self.dimension)

        model_eval = list()
        for i in range(values.shape[0]):
            # Write each value of UQpyOut.txt into a *.txt file
            with open('UQpy_run_{0}.txt'.format(i), 'wb') as f:
                np.savetxt(f, values[i, :], fmt='%0.5f')

            # Run the Input_Shell_Script.sh in order to create the input file for the model
            if self.input_script.lower().endswith('.sh'):
                join_input_script = './{0} {1}'.format(self.input_script, i)
                os.system(join_input_script)
            else:
                print('Unrecognized type of Input file')
                sys.exit()

            # Run the Model.sh in order to run the model
            if self.model_script.lower().endswith('.sh'):
                join_model_script = './{0} {1}'.format(self.model_script, i)
                os.system(join_model_script)
            else:
                print('Unrecognized type of model file')
                sys.exit()

            # Run the Output_Shell_Script.sh  in order to create the input file of the model for UQpy
            if self.output_script.lower().endswith('.sh'):
                join_output_script = './{0} {1}'.format(self.output_script, i)
                os.system(join_output_script)
            else:
                print('Unrecognized type of Input file')
                sys.exit()

            model_eval.append(np.loadtxt('UQpy_eval_{}.txt'.format(i)))

            src_files = 'UQpy_eval_{0}.txt'.format(int(i))
            file_new = src_files.replace("UQpy_eval_{0}.txt".format(int(i)), "Model_{0}.txt".format(int(i)))
            os.rename(src_files, file_new)

        return model_eval

    def run_parallel_model(self, args, multi=False, queue=0):
        import os
        from multiprocessing import Lock
        j = args

        # Define the executable shell scripts for the model

        # Load the UQpyOut.txt
        values = np.loadtxt('UQpy_Batch_{0}.txt'.format(j+1), dtype=np.float32)
        index_temp = np.loadtxt('UQpy_Batch_index_{0}.txt'.format(j + 1))

        index = list()
        for i in range(index_temp.size):
            if index_temp.size == 1:
                index.append(index_temp)
            else:
                index.append(index_temp[i])

        if values.size == 1:
            values = values.reshape(1, 1)

        if len(values.shape) == 1 and self.dimension != 1:
            values = values.reshape(1, values.shape[0])
        elif len(values.shape) == 1 and self.dimension == 1:
            values = values.reshape(values.shape[0], 1)

        os.remove('UQpy_Batch_{0}.txt'.format(j+1))
        os.remove('UQpy_Batch_index_{0}.txt'.format(j + 1))

        model_eval = list()
        count = 0
        for i in index:
            lock = Lock()
            lock.acquire()  # will block if lock is already held

            # Write each value of UQpyOut.txt into a *.txt file
            np.savetxt('UQpy_run_{0}.txt'.format(int(i)), values[count, :], newline=' ', delimiter=',', fmt='%0.5f')

            # Run the Input_Shell_Script.sh in order to create the input file for the model
            if self.input_script.lower().endswith('.sh'):
                join_input_script = './{0} {1}'.format(self.input_script, int(i))
                os.system(join_input_script)
            else:
                print('Unrecognized type of Input file')
                sys.exit()

            # Run the Model.sh in order to run the model
            if self.model_script.lower().endswith('.sh'):
                join_model_script = './{0} {1}'.format(self.model_script, int(i))
                os.system(join_model_script)
            else:
                print('Unrecognized type of model file')
                sys.exit()

            # Run the Output_Shell_Script.sh  in order to create the input file of the model for UQpy
            if self.output_script.lower().endswith('.sh'):
                join_output_script = './{0} {1}'.format(self.output_script, int(i))
                os.system(join_output_script)
            else:
                print('Unrecognized type of Input file')
                sys.exit()

            model_eval.append(np.loadtxt('UQpy_eval_{0}.txt'.format(int(i))))

            src_files = 'UQpy_eval_{0}.txt'.format(int(i))
            file_new = src_files.replace("UQpy_eval_{0}.txt".format(int(i)), "Model_{0}.txt".format(int(i)))
            os.rename(src_files, file_new)

            count = count + 1
            lock.release()

        if multi:
            queue.put(model_eval)

        return model_eval

    def multi_core(self):
        from multiprocessing import Process
        from multiprocessing import Queue

        samples = np.loadtxt('UQpy_Samples.txt', dtype=np.float32)

        if samples.shape[0] <= self.CPUs:
            self.CPUs = samples.shape[0]
            print('The number of CPUs used is\n %', samples.shape[0])

        if len(samples.shape) == 1:
            samples = samples.reshape(samples.shape[0], 1)

        chunk_samples_cores(samples, self)

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
        return results


def chunk_samples_cores(samples, args):

    # In case of parallel computing divide the samples into chunks in order to sent to each processor
    chunks = args.CPUs
    if args.Adaptive is True:
        for i in range(args.CPUs):
            np.savetxt('UQpy_Batch_{0}.txt'.format(i+1), samples[range(i-1, i), :],fmt='%0.5f')
            np.savetxt('UQpy_Batch_index_{0}.txt'.format(i+1), np.array(i).reshape(1,))

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
            np.savetxt('UQpy_Batch_{0}.txt'.format(i+1), samples[lines, :],  fmt='%0.5f')
            np.savetxt('UQpy_Batch_index_{0}.txt'.format(i+1), lines)


def chunk_samples_nodes(samples, args):

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

        np.savetxt('UQpy_Batch_{0}.txt'.format(i+1), samples[lines, :],  fmt='%0.5f')
        np.savetxt('UQpy_Batch_index_{0}.txt'.format(i+1), lines)


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

