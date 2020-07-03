import numpy as np

from UQpy.Distributions import Uniform
from UQpy.RunModel import RunModel
from UQpy.SampleMethods import MCS

dist1 = Uniform(loc=15000, scale=10000)
dist2 = Uniform(loc=450000, scale=80000)
dist3 = Uniform(loc=2.0e8, scale=0.5e8)

names_ = ['fc1', 'fy1', 'Es1',  'fc2', 'fy2', 'Es2', 'fc3', 'fy3', 'Es3', 'fc4', 'fy4', 'Es4', 'fc5', 'fy5', 'Es5',
          'fc6', 'fy6', 'Es6']

x = MCS(dist_object=[dist1, dist2, dist3]*6, nsamples=5, random_state=938475)
samples = np.array(x.samples).round(2)


opensees_rc6_model = RunModel(samples=samples, ntasks=5, model_script='opensees_model.py',
                              input_template='import_variables.tcl', var_names=names_, model_object_name="opensees_run",
                              output_script='process_opensees_output.py', output_object_name='read_output')

outputs = opensees_rc6_model.qoi_list
print(outputs)
