from types import MethodType
from typing import Union

import numpy as np
from beartype import beartype

from UQpy.distributions.baseclass import Copula
from UQpy.distributions.baseclass import (
    DistributionContinuous1D,
    DistributionND,
    DistributionDiscrete1D,
)


class JointCopula(DistributionND):
    @beartype
    def __init__(
        self,
        marginals: Union[list[DistributionContinuous1D], list[DistributionDiscrete1D]],
        copula: Copula,
    ):
        """

        :param marginals: list of distribution objects that define the marginals
        :param copula: copula object
        """
        super().__init__()
        self.ordered_parameters = []
        for i, m in enumerate(marginals):
            self.ordered_parameters.extend(
                [key + "_" + str(i) for key in m.ordered_parameters]
            )
        self.ordered_parameters.extend(
            [key + "_c" for key in copula.ordered_parameters]
        )

        # Check and save the marginals
        self.marginals = marginals
        if not (
            isinstance(self.marginals, list)
            and all(
                isinstance(d, (DistributionContinuous1D, DistributionDiscrete1D))
                for d in self.marginals
            )
        ):
            raise ValueError(
                "Input marginals must be a list of 1d continuous Distribution objects."
            )

        # Check the copula. Also, all the marginals should have a cdf method
        self.copula = copula
        if not isinstance(self.copula, Copula):
            raise ValueError("The input copula should be a Copula object.")
        if not all(hasattr(m, "cdf") for m in self.marginals):
            raise ValueError(
                "All the marginals should have a cdf method in order to define a joint with copula."
            )
        Copula.check_marginals(marginals=self.marginals)

        # Check if methods should exist, if yes define them bound them to the object
        if hasattr(self.copula, "evaluate_cdf"):

            def joint_cdf(dist, x):
                x = dist.check_x_dimension(x)
                # Compute cdf of independent marginals
                unif = np.array(
                    [marg.cdf(x[:, ind_m]) for ind_m, marg in enumerate(dist.marginals)]
                ).T
                # Compute copula
                cdf_val = dist.copula.evaluate_cdf(unit_uniform_samples=unif)
                return cdf_val

            self.cdf = MethodType(joint_cdf, self)

        if all(hasattr(m, "pdf") for m in self.marginals) and hasattr(
            self.copula, "evaluate_pdf"
        ):

            def joint_pdf(dist, x):
                x = dist.check_x_dimension(x)
                # Compute pdf of independent marginals
                pdf_val = np.prod(
                    np.array(
                        [
                            marg.pdf(x[:, ind_m])
                            for ind_m, marg in enumerate(dist.marginals)
                        ]
                    ),
                    axis=0,
                )
                # Add copula term
                unif = np.array(
                    [marg.cdf(x[:, ind_m]) for ind_m, marg in enumerate(dist.marginals)]
                ).T
                c_ = dist.copula.evaluate_pdf(unit_uniform_samples=unif)
                return c_ * pdf_val

            self.pdf = MethodType(joint_pdf, self)

        if all(hasattr(m, "log_pdf") for m in self.marginals) and hasattr(
            self.copula, "evaluate_pdf"
        ):

            def joint_log_pdf(dist, x):
                x = dist.check_x_dimension(x)
                # Compute pdf of independent marginals
                logpdf_val = np.sum(
                    np.array(
                        [
                            marg.log_pdf(x[:, ind_m])
                            for ind_m, marg in enumerate(dist.marginals)
                        ]
                    ),
                    axis=0,
                )
                # Add copula term
                unif = np.array(
                    [marg.cdf(x[:, ind_m]) for ind_m, marg in enumerate(dist.marginals)]
                ).T
                c_ = dist.copula.evaluate_pdf(unit_uniform_samples=unif)
                return np.log(c_) + logpdf_val

            self.log_pdf = MethodType(joint_log_pdf, self)

    def get_parameters(self) -> dict:
        """
        Return the parameters of a :class:`.Distributions` object.

        To update the parameters of a :class:`.JointIndependent` or a :class:`.JointCopula` distribution, each parameter
        is assigned a unique string identifier as :code:`key_index` - where :code:`key` is the parameter name and
        :code:`index` the index of the marginal (e.g., location parameter of the 2nd marginal is identified as
        :code:`loc_1`).

        :return: Parameters of the distribution.
        """
        params = {}
        for i, m in enumerate(self.marginals):
            for key, value in m.get_parameters().items():
                params[key + "_" + str(i)] = value
        for key, value in self.copula.get_parameters().items():
            params[key + "_c"] = value
        return params

    def update_parameters(self, **kwargs: dict):
        """
        Update the parameters of a :class:`.Distributions` object.

        To update the parameters of a :class:`.JointIndependent` or a :class:`.JointCopula` distribution, each
        parameter is assigned a unique string identifier as :code:`key_index` - where :code:`key` is the parameter name
        and :code:`index` the index of the marginal (e.g., location parameter of the 2nd marginal is identified as
        :code:`loc_1`).

        :param kwargs: Parameters to be updated
        :raises ValueError: if *kwargs* contains key that does not already exist.
        """
        # check arguments
        all_keys = self.get_parameters().keys()
        # update the marginal parameters
        for key_indexed, value in kwargs.items():
            if key_indexed not in all_keys:
                raise ValueError("Unrecognized keyword argument " + key_indexed)
            key_split = key_indexed.split("_")
            key, index = "_".join(key_split[:-1]), key_split[-1]
            if index == "c":
                self.copula.parameters[key] = value
            else:
                self.marginals[int(index)].parameters[key] = value
