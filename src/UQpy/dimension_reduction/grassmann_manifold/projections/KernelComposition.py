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
   """
    #: The matrix of left eigenvectors will be used for creating a Grassmann kernel.
    LEFT = 1
    #: The matrix of right eigenvectors will be used for creating a Grassmann kernel.
    RIGHT = 2
    #: The kernel will result from the product of the two above kernels.
    PRODUCT = 3  # doc:
    #: The kernel will result from the summation of the two above kernels.
    SUM = 4
