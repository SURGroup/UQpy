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
                loc = params[0]
                scale = params[1] - params[0]
                return stats.uniform.pdf(x, loc=loc, scale=scale)
            self.pdf = partial(pdf)

            def cdf(x, params):
                loc = params[0]
                scale = params[1] - params[0]
                return stats.uniform.cdf(x, loc=loc, scale=scale)
            self.cdf = partial(cdf)

            def icdf(x, params):
                loc = params[0]
                scale = params[1] - params[0]
                return stats.uniform.ppf(x, loc=loc, scale=scale)
            self.icdf = partial(icdf)

            def moments(params):

                import numpy as np
                y = [np.nan, np.nan, np.nan, np.nan]
                y[0] = (params[0]+params[1])/2
                y[1] = np.sqrt(1/12*(params[1]-params[0])**2)
                return y

            self.moments = partial(moments)

        elif self.name == 'Binomial':

            def pdf(x, params):
                return stats.binom.pdf(x, params[0], params[1])
            self.pdf = partial(pdf)

            def cdf(x, params):
                return stats.binom.cdf(x, params[0], params[1])
            self.cdf = partial(cdf)

            def icdf(x, params):
                return stats.binom.ppf(x, params[0], params[1])
            self.icdf = partial(icdf)

            def moments(params):

                import numpy as np
                y = [np.nan, np.nan, np.nan, np.nan]
                y[0] = params[0]*params[1]
                y[1] = np.sqrt(params[0]*params[1]*(1-params[1]))
                return y

            self.moments = partial(moments)

        elif self.name == 'Beta':

            def pdf(x, params):
                return stats.beta.pdf(x, a=params[0], b=params[1])
            self.pdf = partial(pdf)

            def cdf(x, params):
                return stats.beta.cdf(x, a=params[0], b=params[1])
            self.cdf = partial(cdf)

            def icdf(x, params):
                return stats.beta.ppf(x, a=params[0], b=params[1])
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

        elif self.name == 'Chisquare':

            def pdf(x, params):
                return stats.chi2.pdf(x, params[0])
            self.pdf = partial(pdf)

            def cdf(x, params):
                return stats.chi2.cdf(x, params[0])
            self.cdf = partial(cdf)

            def icdf(x, params):
                return stats.chi2.ppf(x, params[0])
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
                import numpy as np
                return stats.lognorm.pdf(x, s=params[1], scale=np.exp(params[0]))
            self.pdf = partial(pdf)

            def cdf(x, params):
                import numpy as np
                return stats.lognorm.cdf(x, s=params[1], scale=np.exp(params[0]))
            self.cdf = partial(cdf)

            def icdf(x, params):
                import numpy as np
                return stats.lognorm.ppf(x, s=params[1], scale=np.exp(params[0]))
            self.icdf = partial(icdf)

            def moments(params):

                import numpy as np
                y = [np.nan, np.nan, np.nan, np.nan]
                y[0] = np.exp(params[0])
                y[1] = params[1]
                return y

            self.moments = partial(moments)

        elif self.name == 'Gamma':

            def pdf(x, params):
                return stats.gamma.pdf(x, a=params[0], scale=1/params[1])
            self.pdf = partial(pdf)

            def cdf(x, params):
                return stats.gamma.cdf(x,  a=params[0], scale=1/params[1])
            self.cdf = partial(cdf)

            def icdf(x, params):
                return stats.gamma.ppf(x,  a=params[0], scale=1/params[1])
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
                return stats.expon.pdf(x, params[0], scale=1/params[0])
            self.pdf = partial(pdf)

            def cdf(x, params):
                return stats.expon.cdf(x, params[0], scale=1/params[0])
            self.cdf = partial(cdf)

            def icdf(x, params):
                return stats.expon.ppf(x, params[0], scale=1/params[0])
            self.icdf = partial(icdf)

        elif os.path.isfile('custom_dist.py') is True:
            import custom_dist
            self.pdf = getattr(custom_dist, 'pdf')
            self.cdf = getattr(custom_dist, 'cdf')
            self.icdf = getattr(custom_dist, 'icdf')
            self.moments = getattr(custom_dist, 'moments')

