import numpy as np


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
        ################################################################################################################
        # General parameters
        if title == 'Method':
            mydict[title] = lines[lines_[i]+1][:-1]
            print()
        elif title == 'Probability distribution (pdf)':
            dist = []
            j = 0
            while j >= 0:
                testline = lines[lines_[i] + j + 1].strip()
                if not testline:
                    break
                else:
                    dist.append(lines[lines_[i] + j + 1][:-1])
                    j = j + 1
            mydict[title] = dist
        elif title == 'Names of random variables':
            names = []
            j = 0
            while j >= 0:
                testline = lines[lines_[i] + j + 1].strip()
                if not testline:
                    break
                else:
                    names.append(lines[lines_[i] + j + 1][:-1])
                    j = j + 1
            mydict[title] = names
        elif title == 'Probability distribution parameters':
            params = []
            j = 0
            while j >= 0:
                testline = lines[lines_[i] + j + 1].strip()
                if not testline:
                    break
                else:
                    x = lines[lines_[i] + j + 1]
                    params.append(np.float32(x.split(" ")))
                    j = j + 1
            mydict[title] = params
        elif title == 'Number of Samples':
            mydict[title] = int(lines[lines_[i] + 1][:-1])
        elif title == 'Number of random variables':
            mydict[title] = int(lines[lines_[i] + 1][:-1])
        ################################################################################################################
        # Latin Hypercube parameters
        elif title == 'LHS criterion':
            mydict[title] = lines[lines_[i] + 1][:-1]
        elif title == 'distance metric':
            mydict[title] = lines[lines_[i] + 1][:-1]
        elif title == 'iterations':
            mydict[title] = lines[lines_[i] + 1][:-1]
        ################################################################################################################
        #  partially stratified sampling
        elif title == 'PSS design':
            pss_design = []
            j = 0
            while j >= 0:
                testline = lines[lines_[i] + j + 1].strip()
                if not testline:
                    break
                else:
                    x = lines[lines_[i] + j + 1]
                    pss_design.append(int(lines[lines_[i] + j + 1][:-1]))
                    j = j + 1
            mydict[title] = pss_design
        elif title == 'PSS strata':
            pss_strata = []
            j = 0
            while j >= 0:
                testline = lines[lines_[i] + j + 1].strip()
                if not testline:
                    break
                else:
                    pss_strata.append(int(lines[lines_[i] + j + 1][:-1]))
                    j = j + 1
            mydict[title] = pss_strata
        ################################################################################################################
        #  stratified sampling
        elif title == 'STS design':
            sts_design = []
            j = 0
            while j >= 0:
                testline = lines[lines_[i] + j + 1].strip()
                if not testline:
                    break
                else:
                    sts_design.append(int(lines[lines_[i] + j + 1][:-1]))
                    j = j + 1
            mydict[title] = sts_design
            print(sts_design)
        ################################################################################################################
        # Markov Chain Monte Carlo simulation
        elif title == 'MCMC algorithm':
            mydict[title] = lines[lines_[i] + 1][:-1]
        elif title == 'Proposal distribution':
            mydict[title] = lines[lines_[i] + 1][:-1]
        elif title == 'Proposal distribution width':
            mydict[title] = int(lines[lines_[i] + 1][:-1])
        elif title == 'Target distribution':
            mydict[title] = lines[lines_[i] + 1][:-1]
        elif title == 'skip':
            mydict[title] = int(lines[lines_[i] + 1][:-1])
        elif title == 'Target distribution parameters':
            x = lines[lines_[i] + 1]
            target_params = np.float32(x.split(" "))
            mydict[title] = target_params
        elif title == 'seed':
            seed = []
            j = 0
            while j >= 0:
                testline = lines[lines_[i] + j + 1].strip()
                if not testline:
                    break
                else:
                    x = lines[lines_[i] + j + 1]
                    seed.append(np.float32(x.split(" ")))
                    j = j + 1
            mydict[title] = seed
        ################################################################################################################
        # Subset Simulation
        elif title == 'Number of Samples per subset':
            mydict[title] = int(lines[lines_[i] + 1][:-1])
        elif title == 'Conditional probability':
            mydict[title] = np.float32(lines[lines_[i] + 1][:-1])
        elif title == 'Limit-state':
            ls = []
            j = 0
            while j >= 0:
                testline = lines[lines_[i] + j + 1].strip()
                if not testline:
                    break
                else:
                    ls.append(np.float32(lines[lines_[i] + j + 1][:-1]))
                    j = j + 1
                mydict[title] = ls
        elif title == 'Failure probability':
            pf = []
            j = 0
            while j >= 0:
                testline = lines[lines_[i] + j + 1].strip()
                if not testline:
                    break
                else:
                    pf.append(np.float32(lines[lines_[i] + j + 1][:-1]))
                    j = j + 1
                mydict[title] = pf
        ################################################################################################################
        # ADD ANY NEW METHOD HERE

        ################################################################################################################
        # ADD ANY NEW METHOD HERE

        ################################################################################################################
        # ADD ANY NEW METHOD HERE


    return mydict
