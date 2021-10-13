from enum import Enum
import numpy as np

penalty_terms = {
    "BIC": lambda number_of_data, n_params: np.log(number_of_data) * n_params,
    "AICc": lambda number_of_data, n_params: 2 * n_params
    + (2 * n_params ** 2 + 2 * n_params) / (number_of_data - n_params - 1),
    "AIC": lambda number_of_data, n_params: 2 * n_params,
}


class InformationTheoreticCriterion(Enum):
    AIC = 1
    BIC = 2
    AICc = 3
