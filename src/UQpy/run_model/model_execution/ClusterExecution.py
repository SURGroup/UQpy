# pragma: no cover
from __future__ import print_function

import math
import sys

import numpy as np
import os
import pickle

try:
    model = None
    samples = None
    samples_per_process = 0
    samples_shape = None
    samples_list = None
    ranges_list = None
    local_ranges = None
    local_samples = None

    cores_per_task = int(sys.argv[1])
    n_new_simulations = int(sys.argv[2])
    n_existing_simulations = int(sys.argv[3])
    cluster_script = str(sys.argv[4])
    
    with open('model.pkl', 'rb') as filehandle:
        model = pickle.load(filehandle)
        
    with open('samples.pkl', 'rb') as filehandle:
        samples = pickle.load(filehandle)

    # Loop over the number of samples and create input files in a folder in current directory
    for i in range(len(samples)):
        work_dir = os.path.join(model.model_dir, "run_" + str(i))
        model._copy_files(work_dir=work_dir)
        new_text = model._find_and_replace_var_names_with_values(samples[i])
        folder_to_write = 'run_' + str(i+n_existing_simulations) + '/InputFiles'
        # Write the new text to the input file
        model._create_input_files(file_name=model.input_template, num=i+n_existing_simulations,
                                  text=new_text, new_folder=folder_to_write)

    # Use model script to perform necessary preprocessing prior to model execution
    for i in range(len(samples)):
        sample = 'sample' # Sample input in original third-party model, though doesn't seem to use it
        model.execute_single_sample(i, sample)
        
    # Run user-provided cluster script--for now, it is assumed the user knows how to
    # tile jobs in the script
    os.system(f"{cluster_script} {cores_per_task} {n_new_simulations} {n_existing_simulations}")

    results = []

    for i in range(len(samples)):
        # Change current working directory to model run directory
        work_dir = os.path.join(model.model_dir, "run_" + str(i))
        # if model.verbose:
        #     print('\nUQpy: Changing to the following directory for output processing:\n' + work_dir)
        os.chdir(work_dir)
        
        output = model._output_serial(i)
        results.append(output)

    # Change back to model directory
    os.chdir(model.model_dir)
    
    with open('qoi.pkl', 'wb') as filehandle:
        pickle.dump(results, filehandle)
        
except Exception as e:
    print(e)

