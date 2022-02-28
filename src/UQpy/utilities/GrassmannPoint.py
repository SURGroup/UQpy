from typing import Annotated

from beartype import beartype
from beartype.vale import Is

from UQpy.utilities.ValidationTypes import Numpy2DFloatArrayOrthonormal, Numpy2DFloatArray


class GrassmannPoint:
    @beartype
    def __init__(self, data: Numpy2DFloatArrayOrthonormal):
        """
        :param data: Matrix representing the point on the Grassmann manifold.
        """
        self._data = data

    @property
    def data(self) -> Numpy2DFloatArray:
        """
        The matrix containing the Grassmann point
        """
        return self._data
