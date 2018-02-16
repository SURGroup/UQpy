from UQpyLibraries.SampleMethods import *
from various.RunModel import RunModel
from various.module_ import def_model, def_target, readfile
import matplotlib.pyplot as plt
from UQpyLibraries.Reliability import ReliabilityMethods

filename = 'input_SuS.txt'
data = readfile(filename)

# extract input data
model = def_model(data['Model'])
method = data['Method']
N = data['Number of Samples per subset']
d = data['Stochastic dimension']
pdf = data['Probability distribution (pdf)']
pdf_params = data['Probability distribution parameters']
pdf_proposal = data['Proposal distribution']
proposal_pdf_params = np.array(data['Proposal distribution parameters'])
proposal_pdf_width = data['Width of proposal distribution']
p0 = data['Conditional probability']
y = data['Failure criterion']
pdf_target = def_target(data['Target distribution'])
marg_pdf_params = np.array(data['Marginal target distribution parameters'])
jump = data['Burn-in samples']
algorithm = data['MCMC algorithm']


current_dir = os.getcwd()
path = os.path.join(os.sep, current_dir, 'results')
os.makedirs(path, exist_ok=True)
os.chdir(path)


'''
Initialize 
1. class Sampling methods
2. class RunModel
'''

sm = SampleMethods(distribution=pdf, dimension=d, parameters=pdf_params, method=method)
rm = RunModel(model=model)

'''
Run code
'''

SuS = ReliabilityMethods.SubsetSimulation(sm, rm, dimension=d, nsamples_per_subset=N, model=model,
                                        MCMC_algorithm=algorithm, proposal=pdf_proposal,
                                        conditional_prob=p0, marginal_params=marg_pdf_params, jump=jump,
                                        proposal_width=proposal_pdf_width, proposal_params=proposal_pdf_params,
                                        target=pdf_target, limit_state=y)

print(SuS.pf)

'''
Plots
'''


subpath = os.path.join(os.sep, path, 'SuS')
os.makedirs(subpath, exist_ok=True)
os.chdir(subpath)

np.savetxt('samples.txt', SuS.xi, delimiter=' ')
np.savetxt('model.txt', SuS.v)

plt.figure()
plt.scatter(SuS.xi[:, 0], SuS.xi[:, 1])
plt.savefig('samples.png')

plt.figure()
n, bins, patches = plt.hist(SuS.v, 50, normed=1, facecolor='g', alpha=0.75)
plt.title('Histogram')
plt.savefig('histogram.png')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(SuS.xi[:, 0], SuS.xi[:, 1], SuS.v, c='r', s=2)
plt.gca().invert_xaxis()
plt.savefig('model.png')


os.chdir(current_dir)

