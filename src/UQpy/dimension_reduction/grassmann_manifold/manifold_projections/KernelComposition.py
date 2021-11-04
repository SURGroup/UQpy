from enum import Enum

CompositionAction = {
    "LEFT": lambda kernel_psi, kernel_phi: kernel_psi,
    "RIGHT": lambda kernel_psi, kernel_phi: kernel_phi,
    "PRODUCT": lambda kernel_psi, kernel_phi: kernel_psi * kernel_phi,
    "SUM": lambda kernel_psi, kernel_phi: kernel_psi + kernel_phi,
}


class KernelComposition(Enum):
    """
    This is an enumeration which is a set of symbolic names (members) bound to unique, constant values. It is used when
    the SVD is used to project points onto the Grassmann manifold.

    Options:

        1. KernelComposition.LEFT --> The matrix of left eigenvectors will be used for creating a Grassmann kernel.

        2. KernelComposition.RIGHT --> The matrix of right eigenvectors will be used for creating a Grassmann kernel.

        3. KernelComposition.PRODUCT --> The kernel will result from the product of the two above kernels.

        4. KernelComposition.SUM --> The kernel will result from the summation of the two above kernels.
    """

    LEFT = (1,)
    """The matrix of left eigenvectors will be used for creating a Grassmann kernel."""
    RIGHT = (2,)
    PRODUCT = (3,)
    SUM = 4
