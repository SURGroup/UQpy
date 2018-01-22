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

    parser.add_argument("--workDir", dest="Working_directory", action="store",
                        default=None, help="specify the directory of the input file")

    parser.add_argument("--inputFile", default=None,
                        dest="InputFile", action="store",
                        help="specify *.txt file with probabilistic parameters")

    parser.add_argument("--ModelFile", dest="ModelFile", action="store",
                        default=None, help="specify a bash script file for running the model")

    parser.add_argument("--OutDir", dest="Output_directory", action="store",
                        default=None, help="specify the directory where the results are stored")

    args, unknown = parser.parse_known_args()

    if len(unknown) > 0:
        print("Warning:  The following unrecognized arguments were ignored:")
        print(unknown)

    if len(sys.argv) > 1:
        # Run command line version of UQpy
        if args.InputFile is None:
            print("Error:  No input file was selected")
            sys.exit()

        if args.Output_directory is None:
            folder_name = 'simUQpyOut'
            current_dir = os.getcwd()
            path = os.path.join(os.sep, current_dir, folder_name)
            os.makedirs(path, exist_ok=True)
            print("Warning:  No output directory was defined -- The default is %k", path)
            args.Output_directory = path

        # Import UQpy library
        from UQpyLibraries import UQpy_commandLine

        # Exectute UQpy
        UQpy_commandLine.runCommandLine(args)
