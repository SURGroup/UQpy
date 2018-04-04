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
        if title == 'method':
            mydict[title] = lines[lines_[i]+1][:-1]
            print()
        elif title == 'distribution type':
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
        elif title == 'names of parameters':
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
        elif title == 'distribution parameters':
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
        elif title == 'number of samples':
            mydict[title] = int(lines[lines_[i] + 1][:-1])
        elif title == 'number of parameters':
            mydict[title] = int(lines[lines_[i] + 1][:-1])
        ################################################################################################################
        # Latin Hypercube parameters
        elif title == 'criterion':
            mydict[title] = lines[lines_[i] + 1][:-1]
        elif title == 'distance':
            mydict[title] = lines[lines_[i] + 1][:-1]
        elif title == 'iterations':
            mydict[title] = lines[lines_[i] + 1][:-1]
        ################################################################################################################
        #  partially stratified sampling
        elif title == 'design':
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
        elif title == 'strata':
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
        elif title == 'design':
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
        elif title == 'algorithm':
            mydict[title] = lines[lines_[i] + 1][:-1]
        elif title == 'proposal distribution type':
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
        elif title == 'proposal distribution width':
            dist = []
            j = 0
            while j >= 0:
                testline = lines[lines_[i] + j + 1].strip()
                if not testline:
                    break
                else:
                    dist.append(int(lines[lines_[i] + j + 1][:-1]))
                    j = j + 1
            mydict[title] = dist
        elif title == 'target distribution type':
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
        elif title == 'skip':
            mydict[title] = int(lines[lines_[i] + 1][:-1])
        elif title == 'target distribution parameters':
            target_params = list()
            j = 0
            while j >= 0:
                testline = lines[lines_[i] + j + 1].strip()
                if not testline:
                    break
                else:
                    x = lines[lines_[i] + j + 1]
                    target_params.append(np.float32(x.split(" ")))
                    j = j + 1
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
        # Stochastic Reduced Order Model
        elif title == 'SROM':
            mydict[title] = lines[lines_[i] + 1][:-1]
        elif title == 'moments':
            seed = []
            j = 0
            while j >= 0:
                testline = lines[lines_[i] + j + 1].strip()
                if not testline:
                    break
                else:
                    x = lines[lines_[i] + j + 1][:-1]
                    seed.append(np.float32(x.split(" ")))
                    j = j + 1
            mydict[title] = seed
        elif title == 'error function weights':
            seed = []
            j = 0
            while j >= 0:
                testline = lines[lines_[i] + j + 1].strip()
                if not testline:
                    break
                else:
                    x = lines[lines_[i] + j + 1][:-1]
                    seed.append(np.float32(x.split(" ")))
                    j = j + 1
            mydict[title] = seed
        elif title == 'sample weights':
            seed = []
            j = 0
            while j >= 0:
                testline = lines[lines_[i] + j + 1].strip()
                if not testline:
                    break
                else:
                    x = lines[lines_[i] + j + 1][:-1]
                    seed.append(np.float32(x.split(" ")))
                    j = j + 1
            mydict[title] = seed
        elif title == 'properties to match':
            seed = []
            j = 0
            while j >= 0:
                testline = lines[lines_[i] + j + 1].strip()
                if not testline:
                    break
                else:
                    x = lines[lines_[i] + j + 1][:-1]
                    seed.append(np.float32(x.split(" ")))
                    j = j + 1
            mydict[title] = seed
        ################################################################################################################
        # ADD ANY NEW METHOD HERE

        ################################################################################################################
        # ADD ANY NEW METHOD HERE


    return mydict
