from typing import Union

import scipy.stats as stats
from beartype import beartype

from UQpy.distributions.baseclass import DistributionContinuous1D


class Lognormal(DistributionContinuous1D):

    @beartype
    def __init__(
        self,
        s: Union[None, float, int],
        loc: Union[None, float, int] = 0.0,
        scale: Union[None, float, int] = 1.0,
    ):
        """

        :param s: shape parameter
        :param loc: location parameter
        :param scale: scale parameter
        """
        super().__init__(
            s=s, loc=loc, scale=scale, ordered_parameters=("s", "loc", "scale")
        )
        self._construct_from_scipy(scipy_name=stats.lognorm)

    def __repr__(self):
        s = "{s}"
        if self.parameters["loc"] != 0.0:
            s += ", loc={loc}"
        if self.parameters["scale"] != 1.0:
            s += ", scale={scale}"
        return "Lognormal(" + s.format(**self.parameters) + ")"
