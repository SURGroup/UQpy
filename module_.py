import numpy as np
from functools import partial
from modelist import *


def handle_input_file(filename):

    if filename == 'input_mcs.txt':
        with open(filename, "r") as file:
            r_ = 0
            distribution = []
            parameters = []
            for row in [line.split() for line in file if not line.strip().startswith('#')]:
                if len(row) != 0:
                    if r_ == 0:
                        _model = row[0]
                        r_ = r_ + 1
                    elif r_ == 1:
                        method = row[0]
                        r_ = r_ + 1
                    elif r_ == 2:
                        nsamples = int(row[0])
                        r_ = r_ + 1
                    elif r_ == 3:
                        dimension = int(row[0])
                        r_ = r_ + 1
                    elif 4 <= r_ <= 4+dimension-1:
                        distribution.append(row[0])
                        r_ = r_ + 1
                    elif 4+dimension <= r_ <= 4+2*dimension-1:
                        parameters.append([np.float32(row[0]), np.float32(row[1])])
        parameters = np.array(parameters)
        return _model, method, nsamples, dimension, distribution, parameters

    elif filename == 'input_lhs.txt':
        with open(filename, "r") as file:
            r_ = 0
            distribution = []
            parameters = []
            for row in [line.split() for line in file if not line.strip().startswith('#')]:
                if len(row) != 0:
                    if r_ == 0:
                        _model = row[0]
                        r_ = r_ + 1
                    elif r_ == 1:
                        method = row[0]
                        r_ = r_ + 1
                    elif r_ == 2:
                        nsamples = int(row[0])
                        r_ = r_ + 1
                    elif r_ == 3:
                        dimension = int(row[0])
                        r_ = r_ + 1
                    elif 4 <= r_ <= 4+dimension-1:
                        distribution.append(row[0])
                        r_ = r_ + 1
                    elif 4+dimension <= r_ <= 4+2*dimension-1:
                        parameters.append([np.float32(row[0]), np.float32(row[1])])
                        r_ = r_ + 1
                    elif r_ == 4+2*dimension:
                        lhs_criterion = row[0]
                        r_ = r_ + 1
                    elif r_ == 4+2*dimension + 1:
                        dist_metric = row[0]
                        r_ = r_ + 1
                    elif r_ == 4+2*dimension + 2:
                        iterations = int(row[0])
        parameters = np.array(parameters)

        return _model, method, nsamples, dimension, distribution, parameters, lhs_criterion, dist_metric, iterations

    elif filename == 'input_mcmc.txt':
        with open(filename, "r") as file:
            r_ = 0
            distribution = []
            parameters = []
            x0 = []
            params = []
            for row in [line.split() for line in file if not line.strip().startswith('#')]:
                if len(row) != 0:
                    if r_ == 0:
                        _model = row[0]
                        r_ = r_ + 1
                    elif r_ == 1:
                        method = row[0]
                        r_ = r_ + 1
                    elif r_ == 2:
                        nsamples = int(row[0])
                        r_ = r_ + 1
                    elif r_ == 3:
                        dimension = int(row[0])
                        r_ = r_ + 1
                    elif 4 <= r_ <= 4+dimension-1:
                        distribution.append(row[0])
                        r_ = r_ + 1
                    elif 4+dimension <= r_ <= 4+2*dimension-1:
                        parameters.append([np.float32(row[0]), np.float32(row[1])])
                        r_ = r_ + 1
                    elif 4+2*dimension <= r_ <= 3+3*dimension:
                        x0.append(np.float32(row[0]))
                        r_ = r_ + 1
                    elif 4+3*dimension <= r_ <=3+4*dimension:
                        params.append(np.float32(row[0]))
                        r_ = r_ + 1
                    elif r_ == 3+4*dimension + 1:
                        MCMC_algorithm = row[0]
                        r_ = r_ + 1
                    elif r_ == 3+4*dimension + 2:
                        proposal = row[0]
                        r_ = r_ + 1
                    elif r_ == 3+4*dimension + 3:
                        target = row[0]
                        r_ = r_ + 1
                    elif r_ == 3+4*dimension + 4:
                        jump = int(row[0])

        x0 = np.array(x0)
        parameters = np.array(parameters)
        params = np.array(params)

        return _model, method, nsamples, dimension, distribution, parameters, x0, MCMC_algorithm, params, proposal, target, jump

    elif filename == 'input_pss.txt':
        with open(filename, "r") as file:
            r_ = 0
            distribution = []
            parameters = []
            pss_design = []
            pss_stratum = []
            for row in [line.split() for line in file if not line.strip().startswith('#')]:
                if len(row) != 0:
                    if r_ == 0:
                        _model = row[0]
                        r_ = r_ + 1
                    elif r_ == 1:
                        method = row[0]
                        r_ = r_ + 1
                    elif r_ == 2:
                        r_ = r_ + 1
                    elif r_ == 3:
                        dimension = int(row[0])
                        r_ = r_ + 1
                    elif 4 <= r_ <= 4+dimension-1:
                        distribution.append(row[0])
                        r_ = r_ + 1
                    elif 4+dimension <= r_ <= 4+2*dimension-1:
                        parameters.append([np.float32(row[0]), np.float32(row[1])])
                        r_ = r_ + 1
                    elif 4 + 2 * dimension <= r_ <= 3 + 3 * dimension:
                        pss_design.append(int(row[0]))
                        r_ = r_ + 1
                    elif 4 + 3 * dimension <= r_ <= 3 + 4 * dimension:
                        pss_stratum.append(int(row[0]))
                        r_ = r_ + 1

        parameters = np.array(parameters)
        pss_design = np.array(pss_design)
        pss_stratum = np.array(pss_stratum)
        nsamples = np.prod(pss_design)

        return _model, method, nsamples, dimension, distribution, parameters, pss_design, pss_stratum

    elif filename == 'input_sts.txt':
        with open(filename, "r") as file:
            r_ = 0
            distribution = []
            parameters = []
            sts_input = []
            for row in [line.split() for line in file if not line.strip().startswith('#')]:
                if len(row) != 0:
                    if r_ == 0:
                        _model = row[0]
                        r_ = r_ + 1
                    elif r_ == 1:
                        method = row[0]
                        r_ = r_ + 1
                    elif r_ == 2:
                        r_ = r_ + 1
                    elif r_ == 3:
                        dimension = int(row[0])
                        r_ = r_ + 1
                    elif 4 <= r_ <= 4+dimension-1:
                        distribution.append(row[0])
                        r_ = r_ + 1
                    elif 4+dimension <= r_ <= 4+2*dimension-1:
                        parameters.append([np.float32(row[0]), np.float32(row[1])])
                        r_ = r_ + 1
                    elif 4 + 2 * dimension <= r_ <= 3 + 3 * dimension:
                        sts_input.append(int(row[0]))
                        r_ = r_ + 1

        parameters = np.array(parameters)
        sts_input = np.array(sts_input)
        nsamples = np.prod(sts_input)

        return _model, method, nsamples, dimension, distribution, parameters, sts_input


def def_model(_model):
    if _model == 'model_zabaras':
        model = partial(model_zabaras)
    elif _model == 'model_ko2d':
        model = partial(model_ko2d)

    return model


def def_target(target):
    if target == 'mvnpdf':
        target = partial(mvnpdf)
    elif target == 'normpdf':
        target = partial(normpdf)
    return target