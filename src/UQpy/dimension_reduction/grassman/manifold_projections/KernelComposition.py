from enum import Enum

CompositionAction = {
        'LEFT': lambda kernel_psi, kernel_phi: kernel_psi,
        'RIGHT': lambda kernel_psi, kernel_phi: kernel_phi,
        'PRODUCT': lambda kernel_psi, kernel_phi: kernel_psi * kernel_phi,
        'SUM': lambda kernel_psi, kernel_phi: kernel_psi + kernel_phi
    }


class KernelComposition(Enum):
    LEFT = 1,
    RIGHT = 2,
    PRODUCT = 3,
    SUM = 4
