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

    parser.add_argument("--ModelDirectory", dest="Model_directory", action="store",
                        default=None, help="Specify the location of the model's directory.")

    parser.add_argument("--InputShellScript", default=None,
                        dest="Input_Shell_Script", action="store",
                        help="Specify the name of the shell script  *.sh used to transform  the output of UQpy"
                             " (UQpyOut_*.txt file) into the appropriate model input file ")

    parser.add_argument("--OutputShellScript", default=None,
                        dest="Output_Shell_Script", action="store",
                        help="Specify the name of the shell script  *.sh used to transform  the output of the model"
                             " into the appropriate UQpy input file (UQpyInp_*.txt) ")

    parser.add_argument("--ModelShellScript", dest="Model", action="store",
                        default=None, help="Specify the name of the shell script used for running the model")

    parser.add_argument("--CPUs", dest="CPUs", action="store",
                        default=0, type=int, help="Number of local cores to be used for the analysis")

    parser.add_argument("--ClusterNodes", dest="nodes", action="store",type=int,
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

        if args.Model is not None:
            if args.Input_Shell_Script is None:
                print("Error: Shell scripts for communication between UQpy and the model are required. Type --help"
                      "for more information")
                sys.exit()

            elif args.Output_Shell_Script is None:
                print("Error: Shell scripts for communication between UQpy and the model are required. Type --help"
                      "for more information")
                sys.exit()

        # Create UQpy output directory
        print(args.Model_directory)
        import shutil
        folder_name = 'simUQpyOut'
        current_dir = os.getcwd()
        args.Output_directory = os.path.join(os.sep, current_dir, folder_name)
        if os.path.exists(args.Output_directory):
            shutil.rmtree(args.Output_directory)
        os.makedirs(args.Output_directory, exist_ok=False)

        # Import UQpy library
        from UQpyLibraries import UQpyModules

        # Execute UQpy
        UQpyModules.RunCommandLine(args)
