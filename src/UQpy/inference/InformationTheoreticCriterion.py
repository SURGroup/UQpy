from enum import Enum
import numpy as np

penalty_terms = {
    "BIC": lambda number_of_data, n_params: np.log(number_of_data) * n_params,
    "AICc": lambda number_of_data, n_params: 2 * n_params
    + (2 * n_params ** 2 + 2 * n_params) / (number_of_data - n_params - 1),
    "AIC": lambda number_of_data, n_params: 2 * n_params,
}


class InformationTheoreticCriterion(Enum):
    """
    This is an enumeration which is a set of symbolic names (members) bound to unique, constant values.
    It is used to define the Information Theoretic Criterion that will be utilized for in the
    :class:`.InformationModelSelection`.
    """
    #: Akaike information criterion.
    AIC = 1
    #: Bayesian information criterion.
    BIC = 2
    #: Corrected Akaike information criterion for small data sets.
    AICc = 3
