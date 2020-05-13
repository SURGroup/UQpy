from UQpy.SampleMethods import MCS
from UQpy.RunModel import RunModel
import numpy as np

########################################################################################################################
# This will need to be rewritten when MCS is updated.
x = MCS(dist_name=['uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform', 'uniform'],
        dist_params=[[0.02, 0.06], [0.02, 0.01], [0.02, 0.01], [0.0025, 0.0075], [0.02, 0.06], [0.02, 0.01],
                     [0.02, 0.01], [0.0025, 0.0075]],
        var_names=['x0', 'y0', 'z0', 'R0', 'x1', 'y1', 'z1', 'R1'],
        nsamples=12)
########################################################################################################################

run_ = RunModel(samples=x.samples, ntasks=6, model_script='dyna_script.py', input_template='dyna_input.k',
                var_names=x.var_names,  model_dir='dyna_test', cluster=True, verbose=False, fmt='{:>10.4f}',
                cores_per_task=12)






