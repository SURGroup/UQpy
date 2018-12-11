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

"""This module contains functionality for all the transformations supported in UQpy."""


from UQpy.Utilities import *
from UQpy.Distributions import *

# Authors: Dimitris G.Giovanis, Audrey Olivier
# Last Modified: 12/4/18 by Dimitris G. Giovanis


########################################################################################################################
########################################################################################################################
#                                         Correlate standard normal samples
########################################################################################################################

class Correlate:
    """
    Description:
    A class to correlate standard normal samples ~ N(0, 1) given a correlation matrix.

    Input:
    :param input_samples: An object of a SampleMethods class or an array of standard normal samples ~ N(0, 1).
    :type input_samples: object or ndarray

    :param corr_norm: The correlation matrix of the random variables in the standard normal space.
    :type corr_norm: ndarray

    param: distribution: An object list containing the distributions of the random variables.
                         Each item in the list is an object of the Distribution class (see Distributions.py).
                         The list has length equal to dimension.
    :type distribution: list

    Output:
    :return: Correlate.samples: Set of correlated normal samples.
    :rtype: Correlate.samples: ndarray

    """

    # Authors: Dimitris G.Giovanis
    # Last Modified: 7/4/18 by Michael D. Shields

    def __init__(self, input_samples=None, corr_norm=None, dimension=None):

        # If samples is not an array (It should be an instance of a SampleMethods class)
        if isinstance(input_samples, np.ndarray) is False:
            _dict = {**input_samples.__dict__}
            for k, v in _dict.items():
                setattr(self, k, v)

            self.corr_norm = corr_norm
            self.samples_uncorr = self.samples.copy()

            for i in range(len(self.dist_name)):
                if self.dist_name[i].lower() != 'normal' or self.dist_params[i] != [0, 1]:
                    raise RuntimeError("In order to use class 'Correlate' the random variables should be standard"
                                       "normal")

        # If samples is an array
        else:
            print('Caution: The samples provided must be uncorrelated standard normal random variables.')
            self.samples_uncorr = input_samples
            if dimension is None:
                raise RuntimeError("Dimension must be specified when entering samples as an array.")

            self.dimension = dimension
            self.dist_name = ['normal'] * self.dimension
            self.dist_params = [[0, 1]] * self.dimension
            self.corr_norm = corr_norm
            self.distribution = [None] * self.dimension
            for i in range(self.dimension):
                self.distribution[i] = Distribution(self.dist_name[i])

            if self.corr_norm is None:
                raise RuntimeError("A correlation matrix is required.")

        if np.linalg.norm(self.corr_norm - np.identity(n=self.corr_norm.shape[0])) < 10 ** (-8):
            self.samples = self.samples_uncorr.copy()
        else:
            self.samples = run_corr(self.samples_uncorr, self.corr_norm)


########################################################################################################################
########################################################################################################################
#                                         Decorrelate standard normal samples
########################################################################################################################

class Decorrelate:
    """
        Description:

            A class to decorrelate already correlated normal samples given their correlation matrix.

        Input:
            :param input_samples: An object of type Correlate or an array of correlated N(0,1) samples
            :type input_samples: object or ndarray

            param: distribution: An object list containing the distributions of the random variables.
                                 Each item in the list is an object of the Distribution class (see Distributions.py).
                                 The list has length equal to dimension.
            :type distribution: list

            :param corr_norm: The correlation matrix of the random variables in the standard normal space
            :type corr_norm: ndarray

        Output:
            :return: Decorrelate.samples: Set of uncorrelated normal samples.
            :rtype: Decorrelate.samples: ndarray
    """

    # Authors: Dimitris G.Giovanis
    # Last Modified: 6/24/18 by Dimitris G. Giovanis

    def __init__(self, input_samples=None, corr_norm=None, dimension=None):

        # If samples is not an array (It should be an instance of the Correlate class)
        if isinstance(input_samples, np.ndarray) is False:
            _dict = {**input_samples.__dict__}
            for k, v in _dict.items():
                setattr(self, k, v)

            self.corr_norm = corr_norm
            self.samples_corr = self.samples.copy()

            for i in range(len(self.dist_name)):
                if self.dist_name[i].lower() != 'normal' or self.dist_params[i] != [0, 1]:
                    raise RuntimeError("In order to use class 'Decorrelate' the random variables should be standard "
                                       "normal.")

        # If samples is an array
        else:
            print('Caution: The samples provided must be correlated standard normal random variables.')
            self.samples_corr = input_samples
            if dimension is None:
                raise RuntimeError("Dimension must be specified when entering samples as an array.")
            self.dimension = dimension
            self.dist_name = ['normal'] * self.dimension
            self.dist_params = [[0, 1]] * self.dimension
            self.corr_norm = corr_norm
            self.distribution = [None] * self.dimension
            for i in range(self.dimension):
                self.distribution[i] = Distribution(self.dist_name[i])

            if self.corr_norm is None:
                raise RuntimeError("A correlation matrix is required.")

        if np.linalg.norm(self.corr_norm - np.identity(n=self.corr_norm.shape[0])) < 10 ** (-8):
            self.samples = self.samples_corr
        else:
            self.samples = run_decorr(self.samples_corr, self.corr_norm)


########################################################################################################################
########################################################################################################################
#                                         Inverse Nataf transformation
########################################################################################################################


class InvNataf:
    """
        Description:

            A class to perform the inverse Nataf transformation of samples from N(0, 1) to a user-defined distribution.

        Input:
            :param input_samples: An object of a SampleMethods class containing N(0,1) samples or an array of N(0,1)
                                  samples.
            :type input_samples: object or ndarray

            :param dist_name: A list containing the names of the distributions of the random variables.
                              Distribution names must match those in the Distributions module.
                              If the distribution does not match one from the Distributions module,the user must provide
                              custom_dist.py.
                              The length of the string must be 1 (if all distributions are the same) or equal to
                              dimension.
            :type dist_name: string list

            :param dist_params: Parameters of the distribution.
                                Parameters for each random variable are defined as ndarrays
                                Each item in the list, dist_params[i], specifies the parameters for the corresponding
                                distribution, dist[i].
            :type dist_params: list

            :param corr_norm: The correlation matrix of the random variables in the standard normal space
            :type corr_norm: ndarray

            param: distribution: An object list containing the distributions of the random variables.
                                 Each item in the list is an object of the Distribution class (see Distributions.py).
                                 The list has length equal to dimension.
            :type distribution: list

        Output:
            :return: InvNataf.corr: The distorted correlation matrix of the random variables in the standard space;
            :rtype: InvNataf.corr: ndarray

            :return: InvNataf.samplesN01: An array of N(0,1) samples;
            :rtype: InvNataf.corr: ndarray

            :return: InvNataf.samples: An array of samples following the prescribed distributions;
            :rtype: InvNataf.corr: ndarray

            :return: InvNataf.jacobian: An array containing the Jacobian of the transformation.
            :rtype: InvNataf.jacobian: ndarray

    """

    # Authors: Dimitris G. Giovanis
    # Last Modified: 7/15/18 by Michael D. Shields

    def __init__(self, input_samples=None, corr_norm=None, dist_name=None, dist_params=None, dimension=None):

        # Check if samples is a SampleMethods Object or an array
        if isinstance(input_samples, np.ndarray) is False and input_samples is not None:
            _dict = {**input_samples.__dict__}
            for k, v in _dict.items():
                setattr(self, k, v)

            self.dimension = len(dist_name)
            self.dist_name = dist_name
            self.dist_params = dist_params
            if dist_name is None or dist_params is None:
                if not hasattr(self, 'dist_name') or not hasattr(self, 'dist_params'):
                    raise RuntimeError("In order to use class 'InvNataf' the distributions and their parameters must"
                                       "be provided.")

            # Ensure the dimensions of dist_name are consistent
            if type(self.dist_name).__name__ != 'list':
                self.dist_name = [self.dist_name]
            if len(self.dist_name) == 1 and self.dimension != 1:
                self.dist_name = self.dist_name * self.dimension

            # Ensure that dist_params is a list of length dimension
            if type(self.dist_params).__name__ != 'list':
                self.dist_params = [self.dist_params]
            if len(self.dist_params) == 1 and self.dimension != 1:
                self.dist_params = self.dist_params * self.dimension

            self.distribution = [None] * self.dimension
            for j in range(len(self.dist_name)):
                self.distribution[j] = Distribution(self.dist_name[j])

            if not hasattr(self, 'corr_norm'):
                if corr_norm is None:
                    self.corr_norm = np.identity(n=self.dimension)
                    self.corr = self.corr_norm
                elif corr_norm is not None:
                    self.corr_norm = corr_norm
                    self.corr = correlation_distortion(self.distribution, self.dist_params, self.corr_norm)
            else:
                self.corr = correlation_distortion(self.distribution, self.dist_params, self.corr_norm)

            self.samplesN01 = self.samples.copy()
            self.samples = np.zeros_like(self.samples)

            self.samples, self.jacobian = transform_g_to_ng(self.corr_norm, self.distribution, self.dist_params,
                                                            self.samplesN01)

        elif isinstance(input_samples, np.ndarray):
            self.samplesN01 = input_samples
            if dimension is None:
                raise RuntimeError("UQpy: Dimension must be specified in 'InvNataf' when entering samples as an array.")
            self.dimension = dimension

            self.dist_name = dist_name
            self.dist_params = dist_params
            if self.dist_name is None or self.dist_params is None:
                raise RuntimeError("UQpy: Marginal distributions and their parameters must be specified in 'InvNataf' "
                                   "when entering samples as an array.")

            # Ensure the dimensions of dist_name are consistent
            if type(self.dist_name).__name__ != 'list':
                self.dist_name = [self.dist_name]
            if len(self.dist_name) == 1 and self.dimension != 1:
                self.dist_name = self.dist_name * self.dimension

            # Ensure that dist_params is a list of length dimension
            if type(self.dist_params).__name__ != 'list':
                self.dist_params = [self.dist_params]
            if len(self.dist_params) == 1 and self.dimension != 1:
                self.dist_params = self.dist_params * self.dimension

            self.distribution = [None] * self.dimension
            for j in range(len(self.dist_name)):
                self.distribution[j] = Distribution(self.dist_name[j])

            if corr_norm is None:
                self.corr_norm = np.identity(n=self.dimension)
                self.corr = self.corr_norm
            elif corr_norm is not None:
                self.corr_norm = corr_norm
                self.corr = correlation_distortion(self.distribution, self.dist_params, self.corr_norm)

            self.samples = np.zeros_like(self.samplesN01)

            self.samples, self.jacobian = transform_g_to_ng(self.corr_norm, self.distribution, self.dist_params,
                                                            self.samplesN01)

        elif input_samples is None:
            if corr_norm is not None:
                self.dist_name = dist_name
                self.dist_params = dist_params
                self.corr_norm = corr_norm
                if self.dist_name is None or self.dist_params is None:
                    raise RuntimeError("UQpy: In order to use class 'InvNataf', marginal distributions and their "
                                       "parameters must be provided.")

                if dimension is not None:
                    self.dimension = dimension
                else:
                    self.dimension = len(self.dist_name)

                # Ensure the dimensions of dist_name are consistent
                if type(self.dist_name).__name__ != 'list':
                    self.dist_name = [self.dist_name]
                if len(self.dist_name) == 1 and self.dimension != 1:
                    self.dist_name = self.dist_name * self.dimension

                # Ensure that dist_params is a list of length dimension
                if type(self.dist_params).__name__ != 'list':
                    self.dist_params = [self.dist_params]
                if len(self.dist_params) == 1 and self.dimension != 1:
                    self.dist_params = self.dist_params * self.dimension

                self.distribution = [None] * self.dimension
                for j in range(len(self.dist_name)):
                    self.distribution[j] = Distribution(self.dist_name[j])

                self.corr = correlation_distortion(self.distribution, self.dist_params, self.corr_norm)

            else:
                raise RuntimeError("UQpy: To perform the inverse Nataf transformation without samples, a correlation "
                                   "function 'corr_norm' must be provided.")


########################################################################################################################
########################################################################################################################
#                                         Nataf transformation
########################################################################################################################


class Nataf:
    """
        Description:
            A class to perform the Nataf transformation of samples from a user-defined distribution to N(0, 1).

        Input:
            :param input_samples: An object of type SampleMethods, ndarray and InvNataf
            :type input_samples: object

            :param dist_name: A list containing the names of the distributions of the random variables.
                            Distribution names must match those in the Distributions module.
                            If the distribution does not match one from the Distributions module, the user must provide
                            custom_dist.py.
                            The length of the string must be 1 (if all distributions are the same) or equal to dimension
            :type dist_name: string list

            :param dist_params: Parameters of the distribution
                    Parameters for each random variable are defined as ndarrays
                    Each item in the list, dist_params[i], specifies the parameters for the corresponding distribution,
                        dist[i]
            :type dist_params: list

            param: distribution: An object list containing the distributions of the random variables.
                   Each item in the list is an object of the Distribution class (see Distributions.py).
                   The list has length equal to dimension.
            :type distribution: list

            :param corr The correlation matrix of the random variables in the parameter space.
            :type corr: ndarray

            :param itam_error1:
            :type itam_error1: float

            :param itam_error2:
            :type itam_error1: float

            :param beta:
            :type itam_error1: float


        Output:
            :return: Nataf.corr: The distorted correlation matrix of the random variables in the standard space;
            :rtype: Nataf.corr: ndarray

            :return: Nataf.samplesN01: An array of N(0,1) samples;
            :rtype: Nataf.corr: ndarray

            :return: Nataf.samples: An array of samples following the normal distribution.
            :rtype: Nataf.corr: ndarray

            :return: Nataf.jacobian: An array containing the Jacobian of the transformation.
            :rtype: Nataf.jacobian: ndarray
    """

    # Authors: Dimitris G.Giovanis
    # Last Modified: 4/12/2018 by Dimitris G. Giovanis

    def __init__(self, input_samples=None, dimension=None, corr=None, dist_name=None, dist_params=None, beta=None,
                 itam_error1=None, itam_error2=None):

        # If samples is an instance of a SampleMethods class
        if isinstance(input_samples, np.ndarray) is False and input_samples is not None:
            _dict = {**input_samples.__dict__}
            for k, v in _dict.items():
                setattr(self, k, v)

            # Allow to inherit distribution from samples or the user to specify the distribution
            if dist_name is None or dist_params is None:
                if not hasattr(self, 'dist_name') or not hasattr(self, 'dist_params'):
                    raise RuntimeError("In order to use class 'Nataf' the distributions and their parameters must"
                                       "be provided.")

            # Allow to inherit correlation from samples or the user to specify the correlation
            if corr is None:
                if not hasattr(self, 'corr'):
                    self.corr = np.identity(n=self.dimension)
            else:
                self.corr = corr

            self.distribution = [None] * len(self.dist_name)
            for j in range(len(self.dist_name)):
                self.distribution[j] = Distribution(self.dist_name[j])

            # Check for variables that are  standard normal
            count = 0
            for i in range(len(self.dist_name)):
                if self.dist_name[i].lower() == 'normal' and self.dist_params[i] == [0, 1]:
                    count = count + 1

            if count == len(self.dist_name):  # Case where the variables are all standard normal
                self.samplesN01 = self.samples.copy()
                m, n = np.shape(self.samples)
                self.samples = input_samples
                self.Jacobian = list()
                for i in range(m):
                    self.Jacobian.append(np.identity(n=self.dimension))
                self.corr_norm = self.corr
            else:
                if np.linalg.norm(self.corr - np.identity(n=self.corr.shape[0])) < 10 ** (-8):
                    self.corr_norm = self.corr.copy()
                else:
                    self.corr_norm = itam(self.distribution, self.dist_params, self.corr, beta, itam_error1,
                                          itam_error2)

                self.Jacobian = list()
                self.samplesNG = self.samples.copy()
                self.samples = np.zeros_like(self.samplesNG)

                self.samples, self.jacobian = transform_ng_to_g(self.corr_norm, self.distribution, self.dist_params,
                                                                self.samplesNG)

        # If samples is an array
        elif isinstance(input_samples, np.ndarray):
            if dimension is None:
                raise RuntimeError("UQpy: Dimension must be specified in 'Nataf' when entering samples as an array.")
            self.dimension = dimension
            self.samplesNG = input_samples
            if corr is None:
                raise RuntimeError("UQpy: corr must be specified in 'Nataf' when entering samples as an array.")
            self.corr = corr
            self.dist_name = dist_name
            self.dist_params = dist_params
            if self.dist_name is None or self.dist_params is None:
                raise RuntimeError("UQpy: Marginal distributions and their parameters must be specified in 'Nataf' "
                                   "when entering samples as an array.")

            # Ensure the dimensions of dist_name are consistent
            if type(self.dist_name).__name__ != 'list':
                self.dist_name = [self.dist_name]
            if len(self.dist_name) == 1 and self.dimension != 1:
                self.dist_name = self.dist_name * self.dimension

            # Ensure that dist_params is a list of length dimension
            if type(self.dist_params).__name__ != 'list':
                self.dist_params = [self.dist_params]
            if len(self.dist_params) == 1 and self.dimension != 1:
                self.dist_params = self.dist_params * self.dimension

            self.distribution = [None] * self.dimension
            for j in range(len(self.dist_name)):
                self.distribution[j] = Distribution(self.dist_name[j])

            # Check for variables that are non-Gaussian
            count = 0
            for i in range(len(self.distribution)):
                if self.dist_name[i].lower() == 'normal' and self.dist_params[i] == [0, 1]:
                    count = count + 1

            if count == len(self.distribution):
                self.samples = self.samplesNG.copy()
                self.jacobian = list()
                for i in range(len(self.distribution)):
                    self.jacobian.append(np.identity(n=self.dimension))
                self.corr_norm = self.corr
            else:
                if np.linalg.norm(self.corr - np.identity(n=self.corr.shape[0])) < 10 ** (-8):
                    self.corr_norm = self.corr
                else:
                    self.corr = corr
                    self.corr_norm = itam(self.distribution, self.dist_params, self.corr, beta, itam_error1,
                                          itam_error2)

                self.Jacobian = list()
                self.samples = np.zeros_like(self.samplesNG)

                self.samples, self.jacobian = transform_ng_to_g(self.corr_norm, self.distribution, self.dist_params,
                                                                self.samplesNG)

        # Perform ITAM to identify underlying Gaussian correlation without samples.
        elif input_samples is None:
            if corr is not None:
                self.dist_name = dist_name
                self.dist_params = dist_params
                self.corr = corr
                if self.dist_name is None or self.dist_params is None:
                    raise RuntimeError("UQpy: In order to use class 'Nataf', marginal distributions and their "
                                       "parameters must be provided.")

                if dimension is not None:
                    self.dimension = dimension
                else:
                    self.dimension = len(self.dist_name)

                # Ensure the dimensions of dist_name are consistent
                if type(self.dist_name).__name__ != 'list':
                    self.dist_name = [self.dist_name]
                if len(self.dist_name) == 1 and self.dimension != 1:
                    self.dist_name = self.dist_name * self.dimension

                # Ensure that dist_params is a list of length dimension
                if type(self.dist_params).__name__ != 'list':
                    self.dist_params = [self.dist_params]
                if len(self.dist_params) == 1 and self.dimension != 1:
                    self.dist_params = self.dist_params * self.dimension

                self.distribution = [None] * self.dimension
                for j in range(len(self.dist_name)):
                    self.distribution[j] = Distribution(self.dist_name[j])

                count = 0
                for i in range(len(self.dist_name)):
                    if self.dist_name[i].lower() == 'normal' and self.dist_params[i] == [0, 1]:
                        count = count + 1

                if count == len(self.distribution):
                    self.jacobian = list()
                    for i in range(len(self.distribution)):
                        self.jacobian.append(np.identity(n=self.dimension))
                    self.corr_norm = self.corr
                else:
                    if np.linalg.norm(self.corr - np.identity(n=self.corr.shape[0])) < 10 ** (-8):
                        self.corr_norm = self.corr
                    else:
                        self.corr = corr
                        self.corr_norm = itam(self.distribution, self.dist_params, self.corr, beta, itam_error1,
                                              itam_error2)

            else:
                raise RuntimeError("UQpy: To perform the Nataf transformation without samples, a correlation "
                                   "function 'corr' must be provided.")
