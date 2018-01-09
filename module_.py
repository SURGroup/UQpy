import numpy as np
from functools import partial
from modelist import *


def readfile(filename):
    lines_ = []
    mydict = {}
    count = -1
    for line in open(filename):
        rec = line.strip()
        count = count + 1
        if rec.startswith('#'):
            lines_.append(count)

    f = open(filename)
    lines = f.readlines()

    for i in range(len(lines_)):
        title = lines[lines_[i]][1:-1]
        # General parameters
        if title == 'Method':
            mydict[title] = lines[lines_[i]+1][:-1]
            print()
        elif title == 'Stochastic dimension':
            mydict[title] = int(lines[lines_[i] + 1][:-1])
        elif title == 'Probability distribution (pdf)':
            dist = []
            for k in range(mydict['Stochastic dimension']):
                dist.append(lines[lines_[i]+k+1][:-1])
            mydict[title] = dist
        elif title == 'Probability distribution parameters':
            params = []
            for k in range(mydict['Stochastic dimension']):
                params.append([np.float32(lines[lines_[i]+k+1][0]), np.float32(lines[lines_[i]+k+1][2])])
            mydict[title] = params
        elif title == 'Model':
            mydict[title] = lines[lines_[i] + 1][:-1]
        elif title == 'Number of Samples':
            mydict[title] = int(lines[lines_[i] + 1][:-1])
        # Latin Hypercube parameters
        elif title == 'LHS criterion':
            mydict[title] = lines[lines_[i] + 1][:-1]
        elif title == 'distance metric':
            mydict[title] = lines[lines_[i] + 1][:-1]
        elif title == 'iterations':
            mydict[title] = lines[lines_[i] + 1][:-1]
        # partially stratified sampling
        elif title == 'PSS design':
            pss_design = []
            for k in range(mydict['Stochastic dimension']):
                pss_design.append(int(lines[lines_[i]+k+1][0]))
            mydict[title] = pss_design
        elif title == 'PSS strata':
            pss_strata = []
            for k in range(mydict['Stochastic dimension']):
                pss_strata.append(int(lines[lines_[i]+k+1][:-1]))
            mydict[title] = pss_strata
        # stratified sampling
        elif title == 'STS design':
            sts_design = []
            for k in range(mydict['Stochastic dimension']):
                sts_design.append(int(lines[lines_[i]+k+1][:-1]))
            mydict[title] = sts_design
        # Markov Chain Monte Carlo simulation
        elif title == 'MCMC algorithm':
            mydict[title] = lines[lines_[i] + 1][:-1]
        elif title == 'Proposal distribution':
            mydict[title] = lines[lines_[i] + 1][:-1]
        elif title == 'Proposal distribution parameters':
            proposal_params = []
            for k in range(mydict['Stochastic dimension']):
                proposal_params.append(np.float32(lines[lines_[i]+k+1][0]))
            mydict[title] = proposal_params
        elif title == 'Target distribution':
            mydict[title] = lines[lines_[i] + 1][:-1]
        elif title == 'Burn-in samples':
            mydict[title] = int(lines[lines_[i] + 1][:-1])
        elif title == 'Marginal target distribution parameters':
            marg_params = []
            for k in range(mydict['Stochastic dimension']):
                marg_params.append([np.float32(lines[lines_[i]+k+1][0]), np.float32(lines[lines_[i]+k+1][2])])
            mydict[title] = marg_params
        # Subset Simulation
        elif title == 'Number of Samples per subset':
            mydict[title] = int(lines[lines_[i] + 1][:-1])
        elif title == 'Width of proposal distribution':
            mydict[title] = np.float32(lines[lines_[i] + 1][:-1])
        elif title == 'Conditional probability':
            mydict[title] = np.float32(lines[lines_[i] + 1][:-1])
        elif title == 'Failure criterion':
            mydict[title] = np.float32(lines[lines_[i] + 1][:-1])
        # Stochastic Reduced Order Models (SROM)

    return mydict


def def_model(_model):
    if _model == 'model_zabaras':
        model = partial(model_zabaras)
    elif _model == 'model_ko2d':
        model = partial(model_ko2d)
    elif _model == 'model_reliability':
        model = partial(model_reliability)

    return model


def def_target(target):
    if target == 'mvnpdf':
        target = partial(mvnpdf)
    elif target == 'normpdf':
        target = partial(normpdf)
    elif target == 'marginal':
        target = partial(marginal)
    return target


