from UQpy.Surrogates.PCE.PolyChaosLstsq import PolyChaosLstsq
from UQpy.Surrogates.PCE.PolyChaosRidge import PolyChaosRidge
from UQpy.Surrogates.PCE.PolyChaosLasso import PolyChaosLasso


class PCE:
    """
    Constructs a surrogate model based on the Polynomial Chaos Expansion (PCE)
    method.

    **Inputs:**

    * **method** (class):
        object for the method used for the calculation of the PCE coefficients.

    **Methods:**

    """

    def __init__(self, method, verbose=False):
        self.method = method
        self.verbose = verbose
        self.C = None
        self.b = None

    def fit(self, x, y):
        """
        Fit the surrogate model using the training samples and the
        corresponding model values. This method calls the 'run' method of the
        input method class.

        **Inputs:**

        * **x** (`ndarray`):
            `ndarray` containing the training points.

        * **y** (`ndarray`):
            `ndarray` containing the model evaluations at the training points.

        **Output/Return:**

        The ``fit`` method has no returns and it creates an `ndarray` with the
        PCE coefficients.
        """

        if self.verbose:
            print('UQpy: Running PCE.fit')

        if type(self.method) == PolyChaosLstsq:
            self.C = self.method.run(x, y)

        elif type(self.method) == PolyChaosLasso or \
                type(self.method) == PolyChaosRidge:
            self.C, self.b = self.method.run(x, y)

        if self.verbose:
            print('UQpy: PCE fit complete.')

    def predict(self, x_test):

        """
        Predict the model response at new points.
        This method evaluates the PCE model at new sample points.

        **Inputs:**

        * **x_test** (`ndarray`):
            Points at which to predict the model response.

        **Outputs:**

        * **y** (`ndarray`):
            Predicted values at the new points.

        """

        a = self.method.poly_object.evaluate(x_test)

        if type(self.method) == PolyChaosLstsq:
            y = a.dot(self.C)

        elif type(self.method) == PolyChaosLasso or \
                type(self.method) == PolyChaosRidge:
            y = a.dot(self.C) + self.b

        return y
