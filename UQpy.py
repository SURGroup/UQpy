<<<<<<< HEAD
from SampleMethods import *
from RunModel import RunModel
from module_ import handle_input_file, def_model, def_target
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


filename = sys.argv[1]

current_dir = os.getcwd()

path = os.path.join(os.sep, current_dir, 'examples')
os.makedirs(path, exist_ok=True)
os.chdir(path)


if filename == 'input_mcmc.txt':
    _model, method, nsamples, dimension, distribution, parameters, x0, MCMC_algorithm, params,proposal, target, jump = handle_input_file(filename)
    target = def_target(target)

elif filename == 'input_lhs.txt':
    _model, method, nsamples, dimension, distribution, parameters, lhs_criterion, dist_metric,iterations = handle_input_file(filename)

elif filename == 'input_mcs.txt':
    _model, method, nsamples, dimension, distribution, parameters = handle_input_file(filename)

elif filename == 'input_pss.txt':
    _model, method, nsamples, dimension, distribution, parameters, pss_design, pss_stratum=handle_input_file(filename)

elif filename == 'input_sts.txt':
    _model, method, nsamples, dimension, distribution, parameters, sts_input = handle_input_file(filename)

os.chdir(current_dir)

model = def_model(_model)
sm = SampleMethods(dimension=dimension, distribution=distribution, parameters=parameters, method=method)

path = os.path.join(os.sep, current_dir, 'results')
os.makedirs(path, exist_ok=True)
os.chdir(path)

if method == 'mcs':
    g = RunModel(generator=sm,   nsamples=nsamples,  method=method,  model=model)
    subpath = os.path.join(os.sep, path, 'mcs')

elif method == 'lhs':
    g = RunModel(generator=sm,  nsamples=nsamples,  method=method,  model=model, lhs_criterion='random')
    subpath = os.path.join(os.sep, path, 'lhs')

elif method == 'mcmc':
    g = RunModel(generator=sm, nsamples=nsamples, method=method, model=model,  x0=x0, MCMC_algorithm=MCMC_algorithm, proposal=proposal, params=params, target=target, jump=jump)
    subpath = os.path.join(os.sep, path, 'mcmc')

elif method == 'pss':
    g = RunModel(generator=sm,  method=method, model=model,   pss_design=pss_design, pss_stratum=pss_stratum)
    subpath = os.path.join(os.sep, path, 'pss')

elif method == 'sts':
    g = RunModel(generator=sm,  method=method, model = model, sts_input=sts_input)
    subpath = os.path.join(os.sep, path, 'sts')

os.makedirs(subpath, exist_ok=True)
os.chdir(subpath)

np.savetxt('samples.txt', g.samples, delimiter=' ')
np.savetxt('model.txt', g.eval)

plt.figure()
plt.scatter(g.samples[:, 0], g.samples[:, 1])
plt.savefig('samples.png')

plt.figure()
n, bins, patches = plt.hist(g.eval, 50, normed=1, facecolor='g', alpha=0.75)
plt.title('Histogram')
plt.savefig('histogram.png')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(g.samples[:, 0], g.samples[:, 1], g.eval, c='r', s=2)
plt.gca().invert_xaxis()
plt.savefig('model.png')


os.chdir(current_dir)



=======
import os
import sys
from argparse import ArgumentParser

UQpy_commandLine = \
UQpy_commandLine_Usage = """python UQpy.py --{options}"""

if __name__ == '__main__':

    UQpymainDir = os.path.realpath(__file__)
    UQpydirectory = os.path.dirname(UQpymainDir)

    parser = ArgumentParser(description=UQpy_commandLine,
                            usage=UQpy_commandLine_Usage)

    parser.add_argument("--dir", dest="Model_directory", action="store",
                        default=None, help="Specify the location of the model's directory.")

    parser.add_argument("--input", default=None,
                        dest="Input_Shell_Script", action="store",
                        help="Specify the name of the shell script  *.sh used to transform  the output of UQpy"
                             " (UQpyOut_*.txt file) into the appropriate model input file ")

    parser.add_argument("--output", default=None,
                        dest="Output_Shell_Script", action="store",
                        help="Specify the name of the shell script  *.sh used to transform  the output of the model"
                             " into the appropriate UQpy input file (UQpyInp_*.txt) ")

    parser.add_argument("--model", dest="Solver", action="store",
                        default=None, help="Specify the name of the shell script used for running the model")

    parser.add_argument("--CPUs", dest="CPUs", action="store",
                        default=0, type=int, help="Number of local cores to be used for the analysis")

    parser.add_argument("--ClusterNodes", dest="nodes", action="store", type=int,
                        default=1, help="Number of nodes to distribute the model evaluations in "
                        "case of a cluster.")

    args, unknown = parser.parse_known_args()

    if len(unknown) > 0:
        print("Warning:  The following unrecognized arguments were ignored:")
        print(unknown)

    if len(sys.argv) > 1:
        if args.Model_directory is None:
            print("Error: A model directory needs to be specified")
            sys.exit()

        if args.Solver is not None:
            if args.Input_Shell_Script is None:
                print("Error: Shell scripts for communication between UQpy and the model are required. Type --help"
                      "for more information")
                sys.exit()

            elif args.Output_Shell_Script is None:
                print("Error: Shell scripts for communication between UQpy and the model are required. Type --help"
                      "for more information")
                sys.exit()

        # Check number of available cores
        if args.CPUs != 0:
            args.ParallelProcessing = True
            import multiprocessing
            n_cpu = multiprocessing.cpu_count()
            if args.CPUs > n_cpu:
                print("Error: You have available {0:1d} CPUs. Start parallel computing using {0:1d} CPUs".format(n_cpu))
                args.CPUs = n_cpu
        else:
            args.ParallelProcessing = False

        # Create UQpy output directory
        import shutil
        folder_name = 'simUQpyOut'
        current_dir = os.getcwd()
        args.Output_directory = os.path.join(os.sep, current_dir, folder_name)

        if os.path.exists(args.Output_directory):
            shutil.rmtree(args.Output_directory)
        os.makedirs(args.Output_directory, exist_ok=False)

        # Check if Output_directory already exists inside Model directory
        path = os.path.join(os.sep, args.Model_directory, folder_name)
        if os.path.exists(path) and os.path.isdir(path):
            shutil.rmtree(path)

        # Import UQpy library
        from UQpyLibraries import UQpyModules

        # Execute UQpy
        UQpyModules.RunCommandLine(args)
>>>>>>> Dev_Dimitris
