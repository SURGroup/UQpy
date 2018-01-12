from SampleMethods import *
from RunModel import RunModel
from module_ import def_model, def_target, README
from Reliability import ReliabilityMethods
from Surrogates import SurrogateModels
import matplotlib.pyplot as plt

filename = sys.argv[1]

current_dir = os.getcwd()
path = os.path.join(os.sep, current_dir, 'examples')
os.makedirs(path, exist_ok=True)
os.chdir(path)

input = README(filename)
method = input.data['Method']

if method == 'mcs':
    nsamples = input.data['Number of Samples']
    dim = input.data['Stochastic dimension']
    pdf = input.data['Probability distribution (pdf)']
    pdf_params = input.data['Probability distribution parameters']

    sm = SampleMethods(distribution=pdf, dimension=dim, parameters=pdf_params, method=method)
    mcs = sm.MCS(sm, nsamples, dim)

    if 'Model' in input.data:
        model = def_model(input.data['Model'])
        rm = RunModel(model=model)
        fx = rm.Evaluate(rm, mcs.samples)

elif method == 'lhs':
    nsamples = input.data['Number of Samples']
    dim = input.data['Stochastic dimension']
    pdf = input.data['Probability distribution (pdf)']
    pdf_params = input.data['Probability distribution parameters']
    lhs_criterion = input.data['LHS criterion']
    lhs_metric = input.data['distance metric']
    lhs_iter = input.data['iterations']

    sm = SampleMethods(distribution=pdf, dimension=dim, parameters=pdf_params, method=method)
    lhs = sm.LHS(sm, dimension=dim, nsamples=nsamples, lhs_criterion=lhs_criterion, lhs_iter=lhs_iter,
                 lhs_metric=lhs_metric)

    if 'Model' in input.data:
        model = def_model(input.data['Model'])
        rm = RunModel(model=model)
        fx = rm.Evaluate(rm, lhs.samples)


elif method == 'pss':
    nsamples = input.data['Number of Samples']
    dim = input.data['Stochastic dimension']
    pdf = input.data['Probability distribution (pdf)']
    pdf_params = input.data['Probability distribution parameters']
    pss_design = input.data['PSS design']
    pss_strata = input.data['PSS strata']

    sm = SampleMethods(distribution=pdf, dimension=dim, parameters=pdf_params, method=method)
    pss = sm.PSS(pss_design=pss_design, pss_strata=pss_strata)

    if 'Model' in input.data:
        model = def_model(input.data['Model'])
        rm = RunModel(model=model)
        fx = rm.Evaluate(rm, pss.samples)


elif method == 'sts':

    nsamples = input.data['Number of Samples']
    dim = input.data['Stochastic dimension']
    pdf = input.data['Probability distribution (pdf)']
    pdf_params = input.data['Probability distribution parameters']
    sts_design = input.data['STS design']

    sm = SampleMethods(distribution=pdf, dimension=dim, parameters=pdf_params, method=method)
    sts = sm.STS(sm, strata=Strata(nstrata=sts_design))

    if 'Model' in input.data:
        model = def_model(input.data['Model'])
        rm = RunModel(model=model)
        fx = rm.Evaluate(rm, sts.samples)

elif method == 'mcmc':
    nsamples = input.data['Number of Samples']
    dim = input.data['Stochastic dimension']
    pdf = input.data['Probability distribution (pdf)']
    pdf_params = input.data['Probability distribution parameters']
    pdf_proposal = input.data['Proposal distribution']
    pdf__proposal_params = np.array(input.data['Proposal distribution parameters'])
    pdf_target = def_target(input.data['Target distribution'])
    pdf_target_parameters = np.array(input.data['Marginal target distribution parameters'])
    mcmc_seed = np.array(input.data['Initial seed'])
    mcmc_algorithm = input.data['MCMC algorithm']
    mcmc_burnIn = input.data['Burn-in samples']

    sm = SampleMethods(distribution=pdf, dimension=dim, parameters=pdf_params, method=method)
    mcmc = sm.MCMC(sm, number=nsamples, pdf_target=pdf_target, mcmc_algorithm=mcmc_algorithm, pdf_proposal=pdf_proposal,
                   pdf_proposal_params=pdf__proposal_params, mcmc_seed=mcmc_seed,
                   pdf_target_params=pdf_target_parameters, mcmc_burnIn=mcmc_burnIn)

    if 'Model' in input.data:
        model = def_model(input.data['Model'])
        rm = RunModel(model=model)
        fx = rm.Evaluate(rm, mcmc.samples)


elif method == 'SuS':
    model = def_model(input.data['Model'])
    nsamples_ss = input.data['Number of Samples per subset']
    dim = input.data['Stochastic dimension']
    pdf = input.data['Probability distribution (pdf)']
    pdf_params = input.data['Probability distribution parameters']
    pdf_proposal = input.data['Proposal distribution']
    pdf_proposal_params = np.array(input.data['Proposal distribution parameters'])
    pdf_proposal_width = input.data['Width of proposal distribution']
    p0_cond = input.data['Conditional probability']
    fail = input.data['Failure criterion']
    pdf_target = def_target(input.data['Target distribution'])
    pdf_target_params = np.array(input.data['Marginal target distribution parameters'])
    mcmc_burnIn = input.data['Burn-in samples']
    mcmc_algorithm = input.data['MCMC algorithm']

    sm = SampleMethods(distribution=pdf, dimension=d, parameters=pdf_params, method=method)
    rm = RunModel(model=model)

    SuS = ReliabilityMethods.SubsetSimulation(sm, rm, dimension=dim, nsamples_per_subset=nsamples_ss, model=model,
                                        mcmc_algorithm=mcmc_algorithm, pdf_proposal=pdf_proposal,
                                        p0_cond=p0_cond, pdf_target_params=pdf_target_params, mcmc_burnIn=mcmc_burnIn,
                                        pdf_proposal_width=pdf_proposal_width, pdf_proposal_params=pdf_proposal_params,
                                        pdf_target=pdf_target, fail=fail)





'''
Test Polynomial Chaos
'''
PC = SurrogateModels.PolynomialChaos(dimension=dimension, input=g.samples[:200, :], output=g.eval[:200], order=2)
test_index = 140
pc_tilde = SurrogateModels.PolynomialChaos.PCpredictor(PC, g.samples[:test_index, :])
error_pc = abs(g.eval[:test_index]-pc_tilde)

'''
Test Gaussian Process
'''

GP = SurrogateModels.GaussianProcess(input=g.samples[:200, :], output=g.eval[:200])

gp_tilde, gp_std = SurrogateModels.GaussianProcess.GPredictor(GP, g.samples[:test_index, :])
error_gp = abs(g.eval[:test_index]-gp_tilde)

print()


os.makedirs(subpath, exist_ok=True)
os.chdir(subpath)

np.savetxt('samples.txt', g.samples, delimiter=' ')
np.savetxt('model.txt', g.eval)

plt.figure()
plt.scatter(g.samples[:, 0], g.samples[:, 1])
plt.savefig('samples.png')

plt.figure()
n, bins, patches = plt.hist(g.eval, 50, normed=1, facecolor='g', alpha=0.75)
plt.title('Histogram')
plt.savefig('histogram.png')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(g.samples[:, 0], g.samples[:, 1], g.eval, c='r', s=2)
plt.gca().invert_xaxis()
plt.savefig('model.png')


os.chdir(current_dir)



