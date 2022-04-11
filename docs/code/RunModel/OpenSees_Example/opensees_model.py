import os

import fire


def opensees_run(index):
    name_before = "import_variables.tcl"
    name_ = "import_variables_" + str(index) + ".tcl"

    command0 = "cp ./InputFiles/import_variables_" + str(index) + ".tcl ./import_variables.tcl" 
    command1 = "module load opensees && OpenSees test.tcl"

    os.system(command0)
    os.system(command1)
    current_dir_problem = os.getcwd()
    path_data = os.path.join(os.sep, current_dir_problem, 'OutputFiles')
    print(path_data)
    os.makedirs(path_data, exist_ok=True)
    command3 = "cp ./node20001.out ./OutputFiles/node20001_" + str(index) + ".out " 
    os.system(command3)


if __name__ == '__main__':
    fire.Fire(opensees_run)

