# UQpy is distributed under the MIT license.
#
# Copyright (C) 2018  -- Michael D. Shields
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""This module contains functionality for all the distribution supported in UQpy."""

import scipy.stats as stats
from functools import partial
import sys
import os


# Authors: Dimitris G.Giovanis, Michael D. Shields
# Last Modified: 6/7/18 by Dimitris G. Giovanis


########################################################################################################################
#        Define the probability distribution of the random parameters
########################################################################################################################


def pdf(dist):
    dir_ = os.getcwd()
    sys.path.insert(0, dir_)
    for i in range(len(dist)):
        if type(dist).__name__ == 'str':
            if dist == 'Uniform':
                def f(x, params):
                    return stats.uniform.pdf(x, params[0], params[1])

                dist = f

            elif dist == 'Normal':
                def f(x, params):
                    return stats.norm.pdf(x, params[0], params[1])

                dist = f

            elif dist == 'Lognormal':
                def f(x, params):
                    return stats.lognorm.pdf(x, params[0], params[1])

                dist = f

            elif dist == 'Weibull':
                def f(x, params):
                    return stats.weibull_min.pdf(x, params[0], params[1])

                dist = f

            elif dist == 'Beta':
                def f(x, params):
                    return stats.weibull_min.pdf(x, params[0], params[1], params[2], params[3])

                dist = f

            elif dist == 'Exponential':
                def f(x, params):
                    return stats.expon.pdf(x, params[0], params[1])

                dist = f

            elif dist == 'Gamma':
                def f(x, params):
                    return stats.gamma.pdf(x, params[0], params[1], params[2])

                dist = f

            elif os.path.isfile('custom_dist.py') is True:
                import custom_dist
                method_to_call = getattr(custom_dist, dist)

                dist = partial(method_to_call)

            else:
                raise NotImplementedError('Unidentified pdf_type')

    return dist


def cdf(dist):
    dir_ = os.getcwd()
    sys.path.insert(0, dir_)
    for i in range(len(dist)):
        if type(dist).__name__ == 'str':
            if dist == 'Uniform':
                def f(x, params):
                    return stats.uniform.cdf(x, params[0], params[1])

                dist = f

            elif dist == 'Normal':
                def f(x, params):
                    return stats.norm.cdf(x, params[0], params[1])

                dist = f

            elif dist == 'Lognormal':
                def f(x, params):
                    return stats.lognorm.cdf(x, params[0], params[1])

                dist = f

            elif dist == 'Weibull':
                def f(x, params):
                    return stats.weibull_min.cdf(x, params[0], params[1])

                dist = f

            elif dist == 'Beta':
                def f(x, params):
                    return stats.weibull_min.cdf(x, params[0], params[1], params[2], params[3])

                dist = f

            elif dist == 'Exponential':
                def f(x, params):
                    return stats.expon.cdf(x, params[0], params[1])

                dist = f

            elif dist == 'Gamma':
                def f(x, params):
                    return stats.gamma.cdf(x, params[0], params[1], params[2])

                dist = f

            elif os.path.isfile('custom_dist.py') is True:
                import custom_dist
                method_to_call = getattr(custom_dist, dist)

                dist = partial(method_to_call)

            else:
                raise NotImplementedError('Unidentified pdf_type')

    return dist


def inv_cdf(dist):
    dir_ = os.getcwd()
    sys.path.insert(0, dir_)
    for i in range(len(dist)):
        if type(dist).__name__ == 'str':
            if dist == 'Uniform':
                def f(x, params):
                    return stats.uniform.ppf(x, params[0], params[1])

                dist = f

            elif dist == 'Normal':
                def f(x, params):
                    return stats.norm.ppf(x, params[0], params[1])

                dist = f

            elif dist == 'Lognormal':
                def f(x, params):
                    return stats.lognorm.ppf(x, params[0], params[1])

                dist = f

            elif dist == 'Weibull':
                def f(x, params):
                    return stats.weibull_min.ppf(x, params[0], params[1])

                dist = f

            elif dist == 'Beta':
                def f(x, params):
                    return stats.weibull_min.ppf(x, params[0], params[1])

                dist = f

            elif dist == 'Exponential':
                def f(x, params):
                    return stats.expon.ppf(x, params[0], params[1])

                dist = f

            elif dist == 'Gamma':
                def f(x, params):
                    return stats.gamma.ppf(x, params[0], params[1], params[2])

                dist = f

            elif os.path.isfile('custom_dist.py') is True:
                print('came here')
                import custom_dist
                method_to_call = getattr(custom_dist, dist)

                dist = partial(method_to_call)

            else:
                raise NotImplementedError('Unidentified pdf_type')

    return dist


class Distribution:
    def __init__(self, name):

        self.name = name

        if self.name == 'Normal':
            def pdf(x, params):
                return stats.norm.pdf(x, params[0], params[1])
            self.pdf = partial(pdf)

            def cdf(x, params):
                return stats.norm.cdf(x, params[0], params[1])
            self.cdf = partial(cdf)

            def icdf(x, params):
                return stats.norm.ppf(x, params[0], params[1])
            self.icdf = partial(icdf)

            def moments(params):

                import numpy as np
                y = [np.nan, np.nan, np.nan, np.nan]
                y[0] = params[0]
                y[1] = params[1]
                return y

            self.moments = partial(moments)

        elif self.name == 'Uniform':
            def pdf(x, params):
                return stats.uniform.pdf(x, params[0], params[1])
            self.pdf = partial(pdf)

            def cdf(x, params):
                return stats.uniform.cdf(x, params[0], params[1])
            self.cdf = partial(cdf)

            def icdf(x, params):
                return stats.uniform.ppf(x, params[0], params[1])
            self.icdf = partial(icdf)

            def moments(params):

                import numpy as np
                y = [np.nan, np.nan, np.nan, np.nan]
                y[0] = (params[0]+params[1])/2
                y[1] = np.sqrt(1/12*(params[1]-params[0])**2)
                return y

            self.moments = partial(moments)

        elif self.name == 'Beta':

            def pdf(x, params):
                if not params[0] > 0 and params[1] > 0 and params[2] < params[3]:
                    raise RuntimeError("The Beta distribution is not defined "
                                       "for your parameters")
                return stats.beta.pdf(x, params[0], params[1], params[2], params[3])
            self.pdf = partial(pdf)

            def cdf(x, params):
                return stats.beta.cdf(x, params[0], params[1], params[2], params[3])
            self.cdf = partial(cdf)

            def icdf(x, params):
                return stats.beta.ppf(x, params[0], params[1], params[2], params[3])
            self.icdf = partial(icdf)

            def moments(params):

                import numpy as np
                y = [np.nan, np.nan, np.nan, np.nan]
                alpha = params[0]
                beta = params[1]
                a = params[2]
                b = params[3]
                y[0] = (alpha*b + beta*a)/(alpha + beta)
                y[1] = np.sqrt((alpha*beta*(b-a))/(alpha + beta)**2*(alpha + beta + 1))
                return y

            self.moments = partial(moments)

        elif self.name == 'Lognormal':

            def pdf(x, params):
                if not params[0] > 0 and params[1] > 0:
                    raise RuntimeError("The Beta distribution is not defined "
                                       "for your parameters")
                return stats.lognorm.pdf(x, params[0], params[1])
            self.pdf = partial(pdf)

            def cdf(x, params):
                return stats.lognorm.cdf(x, params[0], params[1])
            self.cdf = partial(cdf)

            def icdf(x, params):
                return stats.lognorm.ppf(x, params[0], params[1])
            self.icdf = partial(icdf)

            def moments(params):

                import numpy as np
                y = [np.nan, np.nan, np.nan, np.nan]
                y[0] = np.exp(params[0])
                y[1] = params[1]
                return y

            self.moments = partial(moments)

        elif self.name == 'gamma':
            def pdf(x, params):
                if not params[0] > 0 and params[1] > 0:
                    raise RuntimeError("The Gamma distribution is not defined "
                                       "for your parameters")
                return stats.gamma.pdf(x, params[0], params[1])
            self.pdf = partial(pdf)

            def cdf(x, params):
                return stats.gamma.cdf(x, params[0], params[1])
            self.cdf = partial(cdf)

            def icdf(x, params):
                return stats.gamma.ppf(x, params[0], params[1])
            self.icdf = partial(icdf)

            def moments(params):

                import numpy as np
                y = [np.nan, np.nan, np.nan, np.nan]
                y[0] = params[0]/params[1]
                y[1] = np.sqrt(params[0]/params[1]**2)
                return y

            self.moments = partial(moments)

        elif self.name == 'Exponential':
            def pdf(x, params):
                return stats.expon.pdf(x, params[0], params[1])
            self.pdf = partial(pdf)

            def cdf(x, params):
                return stats.expon.cdf(x, params[0], params[1])
            self.cdf = partial(cdf)

            def icdf(x, params):
                return stats.expon.ppf(x, params[0], params[1])
            self.icdf = partial(icdf)

        elif os.path.isfile('custom_dist.py') is True:
            import custom_dist
            self.pdf = getattr(custom_dist, 'pdf')
            self.cdf = getattr(custom_dist, 'cdf')
            self.icdf = getattr(custom_dist, 'icdf')

