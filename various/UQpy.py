from UQpyLibraries.SampleMethods import *
from various.RunModel import RunModel
from various.module_ import def_model, def_target, README
from UQpyLibraries.Reliability import ReliabilityMethods

filename = sys.argv[1]

current_dir = os.getcwd()
path = os.path.join(os.sep, current_dir, 'examples')
os.makedirs(path, exist_ok=True)
os.chdir(path)

Readme = README(filename)
method = Readme.data['Method']

if method == 'mcs':
    nsamples = Readme.data['Number of Samples']
    dim = Readme.data['Stochastic dimension']
    pdf = Readme.data['Probability distribution (pdf)']
    pdf_params = Readme.data['Probability distribution parameters']

    sm = SampleMethods(distribution=pdf, dimension=dim, parameters=pdf_params, method=method)
    mcs = sm.MCS(sm, nsamples, dim)

    if 'Model' in Readme.data:
        model = def_model(Readme.data['Model'])
        rm = RunModel(model=model)
        fx = rm.Evaluate(rm, mcs.samples)

elif method == 'lhs':
    nsamples = Readme.data['Number of Samples']
    dim = Readme.data['Stochastic dimension']
    pdf = Readme.data['Probability distribution (pdf)']
    pdf_params = Readme.data['Probability distribution parameters']
    lhs_criterion = Readme.data['LHS criterion']
    lhs_metric = Readme.data['distance metric']
    lhs_iter = Readme.data['iterations']

    sm = SampleMethods(distribution=pdf, dimension=dim, parameters=pdf_params, method=method)
    lhs = sm.LHS(sm, dimension=dim, nsamples=nsamples, lhs_criterion=lhs_criterion, lhs_iter=lhs_iter,
                 lhs_metric=lhs_metric)

    if 'Model' in Readme.data:
        model = def_model(input.Readme['Model'])
        rm = RunModel(model=model)
        fx = rm.Evaluate(rm, lhs.samples)


elif method == 'pss':
    nsamples = Readme.data['Number of Samples']
    dim = Readme.data['Stochastic dimension']
    pdf = Readme.data['Probability distribution (pdf)']
    pdf_params = Readme.data['Probability distribution parameters']
    pss_design = Readme.data['PSS design']
    pss_strata = Readme.data['PSS strata']

    sm = SampleMethods(distribution=pdf, dimension=dim, parameters=pdf_params, method=method)
    pss = sm.PSS(pss_design=pss_design, pss_strata=pss_strata)

    if 'Model' in Readme.data:
        model = def_model(Readme.data['Model'])
        rm = RunModel(model=model)
        fx = rm.Evaluate(rm, pss.samples)


elif method == 'sts':

    nsamples = Readme.data['Number of Samples']
    dim = Readme.data['Stochastic dimension']
    pdf = Readme.data['Probability distribution (pdf)']
    pdf_params = Readme.data['Probability distribution parameters']
    sts_design = Readme.data['STS design']

    sm = SampleMethods(distribution=pdf, dimension=dim, parameters=pdf_params, method=method)
    sts = sm.STS(sm, strata=Strata(nstrata=sts_design))

    if 'Model' in Readme.data:
        model = def_model(Readme.data['Model'])
        rm = RunModel(model=model)
        fx = rm.Evaluate(rm, sts.samples)

elif method == 'mcmc':
    nsamples = Readme.data['Number of Samples']
    dim = Readme.data['Stochastic dimension']
    pdf = Readme.data['Probability distribution (pdf)']
    pdf_params = Readme.data['Probability distribution parameters']
    pdf_proposal = Readme.data['Proposal distribution']
    pdf__proposal_params = np.array(Readme.data['Proposal distribution parameters'])
    pdf_target = def_target(Readme.data['Target distribution'])
    pdf_target_parameters = np.array(Readme.data['Marginal target distribution parameters'])
    mcmc_seed = np.array(Readme.data['Initial seed'])
    mcmc_algorithm = Readme.data['MCMC algorithm']
    mcmc_burnIn = Readme.data['Burn-in samples']

    sm = SampleMethods(distribution=pdf, dimension=dim, parameters=pdf_params, method=method)
    mcmc = sm.MCMC(sm, number=nsamples, pdf_target=pdf_target, mcmc_algorithm=mcmc_algorithm, pdf_proposal=pdf_proposal,
                   pdf_proposal_params=pdf__proposal_params, mcmc_seed=mcmc_seed,
                   pdf_target_params=pdf_target_parameters, mcmc_burnIn=mcmc_burnIn)

    if 'Model' in Readme.data:
        model = def_model(Readme.data['Model'])
        rm = RunModel(model=model)
        fx = rm.Evaluate(rm, mcmc.samples)


elif method == 'SuS':
    model = def_model(Readme.data['Model'])
    nsamples_ss = Readme.data['Number of Samples per subset']
    dim = Readme.data['Stochastic dimension']
    pdf = Readme.data['Probability distribution (pdf)']
    pdf_params = Readme.data['Probability distribution parameters']
    pdf_proposal = Readme.data['Proposal distribution']
    pdf_proposal_params = np.array(Readme.data['Proposal distribution parameters'])
    pdf_proposal_width = Readme.data['Width of proposal distribution']
    p0_cond = Readme.data['Conditional probability']
    fail = Readme.data['Failure criterion']
    pdf_target = def_target(Readme.data['Target distribution'])
    pdf_target_params = np.array(Readme.data['Marginal target distribution parameters'])
    mcmc_burnIn = Readme.data['Burn-in samples']
    mcmc_algorithm = Readme.data['MCMC algorithm']

    sm = SampleMethods(distribution=pdf, dimension=dim, parameters=pdf_params, method=method)
    rm = RunModel(model=model)

    SuS = ReliabilityMethods.SubsetSimulation(sm, rm, dimension=dim, nsamples_per_subset=nsamples_ss, model=model,
                                        mcmc_algorithm=mcmc_algorithm, pdf_proposal=pdf_proposal,
                                        p0_cond=p0_cond, pdf_target_params=pdf_target_params, mcmc_burnIn=mcmc_burnIn,
                                        pdf_proposal_width=pdf_proposal_width, pdf_proposal_params=pdf_proposal_params,
                                        pdf_target=pdf_target, fail=fail)



'''
PC = SurrogateModels.PolynomialChaos(dimension=dimension, input=g.samples[:200, :], output=g.eval[:200], order=2)
test_index = 140
pc_tilde = SurrogateModels.PolynomialChaos.PCpredictor(PC, g.samples[:test_index, :])
error_pc = abs(g.eval[:test_index]-pc_tilde)


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

'''