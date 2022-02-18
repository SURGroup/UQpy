from abc import ABC

from beartype import beartype

from UQpy.utilities.distances.baseclass.Distance import Distance


class RiemannianDistance(Distance, ABC):
    @staticmethod
    @beartype
    def check_rows(xi, xj):
        if xi.data.shape[0] != xj.data.shape[0]:
            raise ValueError("UQpy: Incompatible dimensions. The matrices must have the same number of rows.")
