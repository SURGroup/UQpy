from types import MethodType
import numpy as np
from UQpy.Distributions.baseclass import DistributionContinuous1D, DistributionND, DistributionDiscrete1D


########################################################################################################################
#        Multivariate Continuous Distributions
########################################################################################################################

class JointInd(DistributionND):
    """
    Define a joint distribution from its independent marginals. ``JointInd`` is a child class of ``DistributionND``.

    **Inputs:**

    * **marginals** (`list`):
        list of ``DistributionContinuous1D`` or ``DistributionDiscrete1D`` objects that define the marginals.

    Such a multivariate distribution possesses the following methods, on condition that all its univariate marginals
    also possess them:

    * ``pdf``, ``log_pdf``, ``cdf``, ``rvs``, ``fit``, ``moments``.

    The parameters of the distribution are only stored as attributes of the marginal objects. However, the
    *get_params* and *update_params* method can still be used for the joint. Note that, for this purpose, each parameter
    of the joint is assigned a unique string identifier as `key_index` - where `key` is the parameter name and `index`
    the index of the marginal (e.g., location parameter of the 2nd marginal is identified as `loc_1`).

    """
    def __init__(self, marginals):
        super().__init__()
        self.order_params = []
        for i, m in enumerate(marginals):
            self.order_params.extend([key + '_' + str(i) for key in m.order_params])

        # Check and save the marginals
        if not (isinstance(marginals, list) and all(isinstance(d, (DistributionContinuous1D, DistributionDiscrete1D))
                                                    for d in marginals)):
            raise ValueError('Input marginals must be a list of Distribution1d objects.')
        self.marginals = marginals

        # If all marginals have a method, the joint has it to
        if all(hasattr(m, 'pdf') or hasattr(m, 'pmf') for m in self.marginals):
            def joint_pdf(dist, x):
                x = dist._check_x_dimension(x)
                # Compute pdf of independent marginals
                pdf_val = np.ones((x.shape[0], ))
                for ind_m in range(len(self.marginals)):
                    if hasattr(self.marginals[ind_m], 'pdf'):
                        pdf_val *= marginals[ind_m].pdf(x[:, ind_m])
                    else:
                        pdf_val *= marginals[ind_m].pmf(x[:, ind_m])
                return pdf_val
            if any(hasattr(m, 'pdf') for m in self.marginals):
                self.pdf = MethodType(joint_pdf, self)
            else:
                self.pmf = MethodType(joint_pdf, self)

        if all(hasattr(m, 'log_pdf') or hasattr(m, 'log_pmf') for m in self.marginals):
            def joint_log_pdf(dist, x):
                x = dist._check_x_dimension(x)
                # Compute pdf of independent marginals
                pdf_val = np.zeros((x.shape[0],))
                for ind_m in range(len(self.marginals)):
                    if hasattr(self.marginals[ind_m], 'log_pdf'):
                        pdf_val += marginals[ind_m].log_pdf(x[:, ind_m])
                    else:
                        pdf_val += marginals[ind_m].log_pmf(x[:, ind_m])
                return pdf_val
            if any(hasattr(m, 'log_pdf') for m in self.marginals):
                self.log_pdf = MethodType(joint_log_pdf, self)
            else:
                self.log_pmf = MethodType(joint_log_pdf, self)

        if all(hasattr(m, 'cdf') for m in self.marginals):
            def joint_cdf(dist, x):
                x = dist._check_x_dimension(x)
                # Compute cdf of independent marginals
                cdf_val = np.prod(np.array([marg.cdf(x[:, ind_m])
                                            for ind_m, marg in enumerate(dist.marginals)]), axis=0)
                return cdf_val
            self.cdf = MethodType(joint_cdf, self)

        if all(hasattr(m, 'rvs') for m in self.marginals):
            def joint_rvs(dist, nsamples=1, random_state=None):
                # Go through all marginals
                rv_s = np.zeros((nsamples, len(dist.marginals)))
                for ind_m, marg in enumerate(dist.marginals):
                    rv_s[:, ind_m] = marg.rvs(nsamples=nsamples, random_state=random_state).reshape((-1,))
                return rv_s
            self.rvs = MethodType(joint_rvs, self)

        if all(hasattr(m, 'fit') for m in self.marginals):
            def joint_fit(dist, data):
                data = dist._check_x_dimension(data)
                # Compute ml estimates of independent marginal parameters
                mle_all = {}
                for ind_m, marg in enumerate(dist.marginals):
                    mle_i = marg.fit(data[:, ind_m])
                    mle_all.update({key+'_'+str(ind_m): val for key, val in mle_i.items()})
                return mle_all
            self.fit = MethodType(joint_fit, self)

        if all(hasattr(m, 'moments') for m in self.marginals):
            def joint_moments(dist, moments2return='mvsk'):
                # Go through all marginals
                if len(moments2return) == 1:
                    return np.array([marg.moments(moments2return=moments2return) for marg in dist.marginals])
                moments_ = [np.empty((len(dist.marginals), )) for _ in range(len(moments2return))]
                for ind_m, marg in enumerate(dist.marginals):
                    moments_i = marg.moments(moments2return=moments2return)
                    for j in range(len(moments2return)):
                        moments_[j][ind_m] = moments_i[j]
                return tuple(moments_)
            self.moments = MethodType(joint_moments, self)

    def get_params(self):
        """
        Return the parameters of a ``Distributions`` object.

        To update the parameters of a ``JointInd`` or a ``JointCopula`` distribution, each parameter is assigned a
        unique string identifier as `key_index` - where `key` is the parameter name and `index` the index of the
        marginal (e.g., location parameter of the 2nd marginal is identified as `loc_1`).

        **Output/Returns:**

        * (`dict`):
            Parameters of the distribution

        """
        params = {}
        for i, m in enumerate(self.marginals):
            params_m = m.get_params()
            for key, value in params_m.items():
                params[key + '_' + str(i)] = value
        return params

    def update_params(self, **kwargs):
        """
        Update the parameters of a ``Distributions`` object.

        To update the parameters of a ``JointInd`` or a ``JointCopula`` distribution, each parameter is assigned a
        unique string identifier as `key_index` - where `key` is the parameter name and `index` the index of the
        marginal (e.g., location parameter of the 2nd marginal is identified as `loc_1`).

        **Input:**

        * keyword arguments:
            Parameters to be updated

        """
        # check arguments
        all_keys = self.get_params().keys()
        # update the marginal parameters
        for key_indexed, value in kwargs.items():
            if key_indexed not in all_keys:
                raise ValueError('Unrecognized keyword argument ' + key_indexed)
            key_split = key_indexed.split('_')
            key, index = '_'.join(key_split[:-1]), int(key_split[-1])
            self.marginals[index].params[key] = value