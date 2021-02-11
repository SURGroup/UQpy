from UQpy.SampleMethods import MCS, LHS
from UQpy.RunModel import RunModel
import numpy as np
import sys
import pickle
import datetime
import shutil
import os
from ARL_utils import *
from UQpy.RunModel import HMS
from ARL_filter import *

# Things to be set for the run
configFile = 'hmsConf.ini'
pointFileName = "Olivier_main_aluminium.k"

run_name = 'alu_Vx[100]Vz[10]_f[0.01]_[init_vel]_stresses_3runs_Oct05'
nsamples = 20
# ntasks = 3
# cores_per_task = 1
# scratch_dir = '/scratch/users/aolivie1@jhu.edu/LSDYNA_UQpy_runs/' + run_name   # directory where runs happen and results
# are saved

# Things to be set for the model
nvoids = 3    # nb of voids
V_x = 100.   # velocity to pull on the material in x directioon, in mm/ms
V_y, V_z = -10., -5.    # typically V_y=2 V_z, can also be set to None (uniaxial case)
porosity_value = 0.01    # initial porosity
ramp_up_velocity = True    # Do you ramp-up the velocity or set an initial velocity to start

# Other inputs that probably will never change
porosity_bounds = None    # bounds on porosity, i.e. (0.005, 0.1) - set to None if porosity_value is set, and vice-versa
radii_bounds = (0.0025, 0.01)    # bounds on the voids radii
bounds_placement = ((0.02, 0.08), (0.02, 0.08), (0.02, 0.03))    # bounds on placement of void centers within the RVE
t1 = 1e-4   # used only when you ramp-up velocity (you ramp up from 0 to t1, then velocity is constant)
scale_to_MPa = 1e-3    # saved outputs are in N and MPa, requires some units conversion from LSDYNA outputs
scale_to_N = 1e-3
refinement_ale = 1     # defines the mesh size for ALE and side plates
refinement_shells = 1
# kfile = 'Olivier_main_aluminium.k'
templatefile = 'template_Olivier_main_aluminium.k'   # this is the template that is always the same, some lines get
# added to create the final kfile


########################################################################################################################
########################################################################################################################


# Get the template file and add necessary lines (BCs, voids etc)
shutil.copyfile(templatefile, pointFileName)
with open(pointFileName, 'a') as f:
    f.write('$\n')
# Add the inputs needed for the voids
# write_geometry_voids(kfile=kfile, nvoids=nvoids)
write_geometry_voids(kfile=pointFileName, nvoids=nvoids)
# Add inputs needed for BCs
write_BCs(kfile=pointFileName, V_x=V_x, V_y=V_y, V_z=V_z, ramp_up_velocity=ramp_up_velocity, t1=t1)
with open(pointFileName, 'a') as f:
    f.write('*END')

# Define the HMS object used to run the model
HMS_model = HMS(exec_prefix='mpiexec -np 1 -machinefile machinefile', exec_path='ls971_d C=36000 memory=400000000 i=',
                resourceType='CPU', resourceAmount=1, hms_pointFileName=pointFileName,
                hms_configFile=configFile)

# Sample and place voids
# var_names = []
# [var_names.extend(['x'+str(i), 'y'+str(i), 'z'+str(i), 'R'+str(i)]) for i in range(nvoids)]
samples = np.zeros((nsamples, 4 * nvoids))
files2copy=['shells_refinement[4].k', 'settings.pkl']
fmt = '{:>10.4f}'
var_names = ['x', 'y', 'z', 'R']

for i in range(nsamples):
    voids_radii = sample_voids_radii(
        n_voids=nvoids, radii_bounds=radii_bounds, porosity_value=porosity_value, porosity_bounds=porosity_bounds)
    voids_xyz, voids_radii = place_voids_no_overlap(voids_radii=voids_radii, bounds_placement=bounds_placement)
    samples[i, :] = np.concatenate([voids_xyz, np.array(voids_radii).reshape((-1, 1))], axis=1).reshape((-1, ))

    argument = LSArgument(nvoids, samples[i, :], i)

    inputFilter = LSInputFilter(pointFileName=pointFileName, var_names=var_names, files2copy=files2copy, fmt=fmt)

    outputFilter = LSOutputFilter()

    # Save settings that will be used within the script to get some info about the run
    ts = datetime.datetime.now().strftime('%Y_%m_%d_%I_%M_%p')
    with open('settings.pkl', 'wb') as f:
        pickle.dump({'nvoids': nvoids, 'run_name': run_name, 'V_x': V_x, 'V_y': V_y, 'V_z': V_z,
                     'ramp_up_velocity': ramp_up_velocity, 't1': t1,
                     'kfile': pointFileName, 'scale_to_MPa': scale_to_MPa, 'scale_to_N': scale_to_N, 'samples': samples,
                     'datetime': ts}, f)

    HMS_model.run(hms_inputFilter=inputFilter, hms_outputFilter=outputFilter, hms_argument=argument)

while None in HMS_model.qoi_list:
    # Pull model results
    HMS_model.receive()

sys.stderr.write(str(HMS_model.qoi_list))
print(HMS_model.qoi_list)

# run_ = RunModel(samples=samples, model_script='ARL_script.py', model_object_name='run_dyna_model', input_template=kfile,
#                 output_script='ARL_output_script.py', var_names=var_names, model_dir=scratch_dir, verbose=True,
#                 fmt='ls-dyna', output_object_name='return_output_none', cluster=True, ntasks=ntasks,
#                 cores_per_task=cores_per_task)

# remove the kfile and seetings.pkl from the LSDYNA file, (just keep the template)
# for file_name in ['settings.pkl', pointFileName]:
#     if os.path.exists(os.path.join(os.getcwd(), file_name)):
#         os.remove(os.path.join(os.getcwd(), file_name))
