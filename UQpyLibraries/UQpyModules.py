import os
import shutil
from UQpyLibraries.SampleMethods import *


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
        # Run UQpy

        print("\nExecuting UQpy...\n")
        ################################################################################################################
        # Run Subset Simulation
        if data['Method'] == 'SuS':
            self.args.Adaptive = True
            from UQpyLibraries.Reliability import SubsetSimulation
            init_sm(data)
            self.SuS = SubsetSimulation(self.args, data)
        elif data['Method'] in ['mcs', 'lhs', 'mcmc', 'pss', 'sts']:
            self.args.Adaptive = False
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

        ################################################################################################################
        # Initialize the requested UQpy method: Check if all necessary parameters are defined in the UQpyParams.txt file
        init_sm(data)

        ################################################################################################################
        # Run the requested UQpy method and save the samples into file 'UQpyOut.txt'
        samples_01 = run_sm(data)

        # Transform samples from U(0, 1) to the original parameter space
        if data['Method'] != 'mcmc':
            samples = transform_pdf(samples_01, data['Probability distribution (pdf)'],
                                    data['Probability distribution parameters'])
        else:
            samples = samples_01

        # Save the samples in a .txt file
        save_txt(data['Names of random variables'], samples)

        # Save the samples in a .csv file
        save_csv(data['Names of random variables'], samples)

        ################################################################################################################
        # Split the samples into chunks in order to sent to each processor in case of parallel computing

        if self.args.ParallelProcessing is True:
            if samples.shape[0] <= self.args.CPUs:
                self.args.CPUs = samples.shape[0]
                self.args.CPUs_flag = True
                print('The number of CPUs used is\n %', samples.shape[0])
            chunk_samples_cores(data, samples, self.args)

        ################################################################################################################
        # If a model is provided then run it

        if self.args.Solver is not None:
            RunModel(self.args)

        ################################################################################################################
        # Move the data to directory simUQpyOut/ , delete the temp/ directory
        # and terminate the program
        _files = []
        _files.append('UQpy_Samples.csv')
        _files.append('UQpy_Samples.txt')

        if self.args.Solver is not None:
            src_files = [filename for filename in os.listdir(self.args.WorkingDir) if filename.startswith("UQpy_eval_")]
            for file in src_files:
                file_new = file.replace("UQpy_eval_", "Model_")
                os.rename(file, file_new)
                _files.append(file_new)

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

    :param args
    """
    def __init__(self, args):

        self.CPUs = args.CPUs
        self.model_script = args.Solver
        self.input_script = args.Input_Shell_Script
        self.output_script = args.Output_Shell_Script
        self.current_dir = os.getcwd()
        self.Adaptive = args.Adaptive
        parallel_processing = args.ParallelProcessing

        if parallel_processing is True:
            self.values = self.multi_core()
        else:
            self.values = self.run_model()

    def run_model(self):
        import time
        start_time = time.time()

        # Load the UQpyOut.txt
        values = np.loadtxt('UQpy_Samples.txt', dtype=np.float32)

        print("\nEvaluating the model...\n")
        model_eval = []

        if self.Adaptive is True:
            values = values.reshape(1, values.shape[1])

        for i in range(values.shape[0]):
            # Write each value of UQpyOut.txt into a *.txt file
            with open('UQpy_run_{0}.txt'.format(i), 'wb') as f:
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

            model_eval.append(np.loadtxt('UQpy_eval_{}.txt'.format(i)))

        end_time = time.time()
        print('Total time:', end_time - start_time, "(sec)")
        return model_eval

    def run_parallel_model(self, args, multi=False, queue=0):
        import os
        from multiprocessing import Lock
        j = args

        # Define the executable shell scripts for the model

        # Load the UQpyOut.txt
        values = np.loadtxt('UQpy_Batch_{0}.txt'.format(j+1), dtype=np.float32)
        index = np.loadtxt('UQpy_Batch_index_{0}.txt'.format(j + 1))

        if index.size == 1:
            index = []
            values = values.reshape(1, values.shape[0])
            index.append(np.loadtxt('UQpy_Batch_index_{0}.txt'.format(j+1)))
        else:
            index = np.loadtxt('UQpy_Batch_index_{0}.txt'.format(j + 1))

        model_eval = []
        count = 0
        for i in index:
            lock = Lock()
            print(index)
            lock.acquire()  # will block if lock is already held

            # Write each value of UQpyOut.txt into a *.txt file
            np.savetxt('UQpy_run_{0}.txt'.format(int(i)), values[count, :], newline=' ', delimiter=',', fmt='%0.5f')

            # Run the Input_Shell_Script.sh in order to create the input file for the model
            join_input_script = './{0} {1}'.format(self.input_script, int(i))
            os.system(join_input_script)

            # Run the Model.sh in order to run the model
            join_model_script = './{0} {1}'.format(self.model_script, int(i))
            os.system(join_model_script)

            # Run the Output_Shell_Script.sh  in order to create the input file of the model for UQpy
            join_output_script = './{0} {1}'.format(self.output_script, int(i))
            os.system(join_output_script)

            model_eval.append(np.loadtxt('UQpy_eval_{0}.txt'.format(int(i)), dtype=np.float32))
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
        print('Total time:', end_time - start_time, "(sec)")
        return results


def chunk_samples_cores(data, samples, args):

    header = ', '.join(data['Names of random variables'])
    # In case of parallel computing divide the samples into chunks in order to sent to each processor
    chunks = args.CPUs
    if args.CPUs_flag is True:
        for i in range(args.CPUs):
            np.savetxt('UQpy_Batch_{0}.txt'.format(i+1), samples[range(i-1, i), :], header=str(header), fmt='%0.5f')
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
            np.savetxt('UQpy_Batch_{0}.txt'.format(i+1), samples[lines, :], header=str(header), fmt='%0.5f')
            np.savetxt('UQpy_Batch_index_{0}.txt'.format(i+1), lines)


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

        np.savetxt('UQpy_Batch_{0}.txt'.format(i+1), samples[lines, :], header=str(header), fmt='%0.5f')
        np.savetxt('UQpy_Batch_index_{0}.txt'.format(i+1), lines)


def save_csv(headers, param_values):

    index = np.array(range(1, param_values.shape[0] + 1)).astype(int)
    param_values = np.hstack((index.reshape(index.shape[0], 1), param_values))
    HEADER=[]
    HEADER.append('Run')
    for i in range(len(headers)):
        HEADER.append(headers[i])
    import csv
    with open('UQpy_Samples.csv', "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerow(HEADER)
        for val in param_values:
            writer.writerow(val)


def save_txt(headers, param_values):

    header = ', '.join(headers)
    np.savetxt('UQpy_Samples.txt', param_values, header=str(header), fmt='%0.5f')