from odbAccess import *
from abaqusConstants import *
from textRepr import *
import timeit
import numpy as np
import os
import sys


start_time = timeit.default_timer()
index = sys.argv[-1]
# print(index)
# index = float(index)
index = int(index)
# print(index)

odbFile = os.path.join(os.getcwd(), "single_element_simulation_" + str(index) + ".odb")
odb = openOdb(path=odbFile)
step1 = odb.steps.values()[0]

his_key = 'Element PART-1-1.1 Int Point 1 Section Point 1'
region = step1.historyRegions[his_key]
LE22 = region.historyOutputs['LE22'].data
S22 = region.historyOutputs['S22'].data
# t = np.array(LE22)[:, 0]
x = np.array(LE22)[:, 1]
y = np.array(S22)[:, 1]

fnm = os.path.join(os.getcwd(), 'Output', 'output_element_{0}.csv'.format(index))
if not os.path.exists(os.path.dirname(fnm)):
    try:
        os.makedirs(os.path.dirname(fnm))
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

output_file = open(fnm, 'wb')
for k in range(len(x)):
    output_file.write('%13.6e, %13.6e\n' % (x[k], y[k]))
output_file.close()

elapsed = timeit.default_timer() - start_time
print('Finished running odb_process_script. It took ' + str(elapsed) + ' s to run.')
