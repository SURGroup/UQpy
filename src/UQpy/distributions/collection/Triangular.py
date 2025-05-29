from scipy import stats
from UQpy.distributions.baseclass import DistributionContinuous1D


class Triangular(DistributionContinuous1D):

    def __init__(self, c: float, loc: float = 0.0, scale: float = 1.0):
        """

        :param c: Shape parameter between :math:`0 \leq c \leq 1`
        :param loc: The non-zero part distribution starts at ``loc``. Default: 0.0
        :param scale: The width of the non-zero part of the distribution. Default: 1.0
        """
        super().__init__(
            c=c, loc=loc, scale=scale, ordered_parameters=("c", "loc", "scale")
        )
        self._construct_from_scipy(scipy_name=stats.triang)

    def __repr__(self):
        s = "c={c}"
        if self.parameters["loc"] != 0.0:
            s += ", loc={loc}"
        if self.parameters["scale"] != 1.0:
            s += ", scale={scale}"
        s = s.format(**self.parameters)
        return f"Triangular({s})"
