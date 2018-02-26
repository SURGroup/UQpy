import numpy as np
import sys

index = sys.argv[1]
filename = 'modelInput_{0}.txt'.format(int(index))
x = np.loadtxt(filename, dtype=np.float32)

P = 100
l = 5
E = x[0]
I = x[1]

d = P*(l**3)/(3*E*I)

#p = np.sqrt(abs(np.sum(x)))

with open('solution_{0}.txt'.format(int(index)), 'w') as f:
    f.write('{} \n'.format(d))
