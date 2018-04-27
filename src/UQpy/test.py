from UQpy.SampleMethods import MCMC
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp
import time

# p = sp.multivariate_normal.pdf([-1.44,0.0],mean=np.zeros(2),cov=np.eye(2))

def Rosenbrock(x):
    return np.exp(-(100*(x[1]-x[0]**2)**2+(1-x[0])**2)/20)

print(Rosenbrock([-0.274,2.12]))
print(Rosenbrock([1.27,4.44]))

def Normal(x):
    return sp.norm.pdf(x)

x = MCMC(dimension=2, pdf_proposal_type='Normal', pdf_proposal_scale=[1.5,2.3], pdf_target_type='joint_pdf',
         pdf_target=Rosenbrock, algorithm='MMH', jump=100, nsamples=100, seed=None)

# plt.plot(x.samples[:,0],x.samples[:,1],'o')
# plt.show()

# y = list(x.samples)

t = time.time()
z = MCMC(dimension=2, pdf_proposal_type='Normal', pdf_proposal_scale=2, pdf_target_type='joint_pdf',
         pdf_target=Rosenbrock, algorithm='Stretch', jump=1000, nsamples=1000, seed=x.samples)
t_stretch = time.time()-t
print(t_stretch)

t2 = time.time()
y = MCMC(dimension=2, pdf_proposal_type='Normal', pdf_proposal_scale=1, pdf_target_type='joint_pdf',
         pdf_target=Rosenbrock, algorithm='MMH', jump=1000, nsamples=1000, seed=None)
t_MMH = time.time()-t2
print(t_MMH)


plt.plot(z.samples[:,0],z.samples[:,1],'o')
plt.plot(y.samples[:,0],y.samples[:,1],'x')
plt.legend(('Stretch','MMH'))
plt.show()
# plt.hist(x.samples)
# plt.show()



# trial = x.pdf_proposal_type * x.dimension

print(x.samples)