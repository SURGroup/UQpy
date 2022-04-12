# pragma: no cover
from __future__ import print_function

import math
import sys

import numpy as np
from mpi4py import MPI
import os
import pickle

try:
    comm = MPI.COMM_WORLD

    model = None
    samples = None
    samples_per_process = 0
    samples_shape = None
    samples_list = None
    ranges_list = None
    local_ranges = None
    local_samples = None
    if comm.rank == 0:
        n_existing_simulations = int(sys.argv[1])
        n_new_simulations = int(sys.argv[2])

        with open('model.pkl', 'rb') as filehandle:
            model = pickle.load(filehandle)

        with open('samples.pkl', 'rb') as filehandle:
            samples = pickle.load(filehandle)

        print(len(samples))
        print(samples[0].shape)

        samples_shape = list(samples.shape)
        n_samples = len(samples)
        samples_per_process = math.floor(n_samples / comm.size if n_samples / comm.size == 0 else n_samples / comm.size + 1)

        samples_list = []
        ranges_list = []
        for i in range(comm.size):
            start_index = samples_per_process * i
            if start_index > n_samples:
                samples_shape[0] = 0
                samples_list.append(np.empty(samples_shape))
                ranges_list.append(range(start_index, start_index))
                continue
            end_index = min(n_samples, samples_per_process * (i + 1))
            samples_list.append(samples.take(indices=range(start_index, end_index), axis=0))
            ranges_list.append(range(start_index, end_index))

    local_ranges = comm.scatter(ranges_list, root=0)
    print(list(local_ranges))
    local_samples = comm.scatter(samples_list, root=0)

    # broadcast model among processes
    model = comm.bcast(model, root=0)

    results = []

    if len(local_ranges) != 0:
        print(f"I am process {comm.rank} out of {comm.size} on node {MPI.Get_processor_name()} and my range is {list(local_ranges)}")
        index_start = local_ranges[0]
        print(index_start)
        for i in local_ranges:
            index = i - local_ranges[0]
            sample = model.preprocess_single_sample(i, sample=local_samples[index])

            execution_output = model.execute_single_sample(i, sample)

            results.append(model.postprocess_single_file(i, execution_output))

    qoi = comm.gather(results, root=0)

    if comm.rank == 0:
        result = []
        [result.extend(el) for el in qoi]
        with open('qoi.pkl', 'wb') as filehandle:
            pickle.dump(result, filehandle)

    comm.Barrier()  # wait for everybody to synchronize _here_

except Exception as e:
    print(e)

