import scipy.stats as stats
from UQpy.distributions.baseclass.Distribution1D import Distribution1D
from abc import ABC


class DistributionContinuous1D(Distribution1D, ABC):
    """
    Parent class for univariate continuous probability distributions.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _construct_from_scipy(self, scipy_name=stats.rv_continuous):
        self.pdf = lambda x: scipy_name.pdf(x=self.check_x_dimension(x), **self.parameters)
        self.log_pdf = lambda x: scipy_name.logpdf(x=self.check_x_dimension(x), **self.parameters)
        self._retrieve_1d_data_from_scipy(scipy_name)

        def tmp_fit(dist, data):
            data = self.check_x_dimension(data)
            fixed_params = {}
            for key, value in dist.parameters.items():
                if value is not None:
                    fixed_params["f" + key] = value
            params_fitted = scipy_name.fit(data=data, **fixed_params)
            return dict(zip(dist.ordered_parameters, params_fitted))

        self.fit = lambda data: tmp_fit(self, data)
