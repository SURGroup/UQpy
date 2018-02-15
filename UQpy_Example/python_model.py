import numpy as np
import sys

index = sys.argv[1]
filename = 'modelInput_{0}.txt'.format(int(index))
x = np.loadtxt(filename, dtype=np.float32)

p = np.sum(x)

with open('solution_{0}.txt'.format(int(index)), 'w') as f:
    f.write('{} \n'.format(p))
