import os
import fire
import numpy as np
import shutil


def run_dyna_model(index):
    index = int(index)
    print(os.getcwd())
    input_file_name = 'dyna_input_' + str(index) + '.k'
    input_file_path = os.path.join(os.getcwd(), 'InputFiles', input_file_name)
    print(input_file_path)
   
    # command = 'mkdir junk'
    command = 'ls-dyna i=' + input_file_path + ' memory=300000000'
    command1 = 'rm d3* adptmp *.inc *.tmp scr* disk* mes* kill* bg*'

    print(command)
    os.system(command) 
    os.system(command1)


if __name__ == '__main__':
    fire.Fire(run_dyna_model)
