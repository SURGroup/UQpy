import numpy as np

class PolyChaosLasso:
    """
    Class to calculate the PCE coefficients with the Least Absolute Shrinkage
    and Selection Operator (LASSO) method.

    **Inputs:**

    * **poly_object** ('class'):
        Object from the 'Polynomial' class

    **Methods:**

    """

    def __init__(self, poly_object, learning_rate=0.01, iterations=1000,
                 penalty=1, verbose=False):
        self.poly_object = poly_object
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.penalty = penalty
        self.verbose = verbose

    def run(self, x, y):
        """
        Implements the LASSO method to compute the PCE coefficients.

        **Inputs:**

        * **poly_object** (`object`):
            Polynomial object.

        * **learning_rate** (`float`):
            Size of steps for the gradient descent.

        * **iterations** (`int`):
            Number of iterations of the optimization algorithm.

        * **penalty** (`float`):
            Penalty parameter controls the strength of regularization. When it
            is close to zero, then the Lasso regression converges to the linear
            regression, while when it goes to infinity, PCE coefficients
            converge to zero.

        **Outputs:**

        * **w** (`ndarray`):
            Returns the weights (PCE coefficients) of the regressor.

        * **b** (`float`):
            Returns the bias of the regressor.
        """

        xx = self.poly_object.evaluate(x)
        m, n = xx.shape

        if y.ndim == 1 or y.shape[1] == 1:
            y = y.reshape(-1, 1)
            w = np.zeros(n).reshape(-1, 1)
            dw = np.zeros(n).reshape(-1, 1)
            b = 0

            for _ in range(self.iterations):
                y_pred = (xx.dot(w) + b)

                for i in range(n):
                    if w[i] > 0:
                        dw[i] = (-(2 * (xx.T[i, :]).dot(y - y_pred)) + self.penalty) / m
                    else:
                        dw[i] = (-(2 * (xx.T[i, :]).dot(y - y_pred)) - self.penalty) / m

                db = - 2 * np.sum(y - y_pred) / m

                w = w - self.learning_rate * dw
                b = b - self.learning_rate * db

        else:
            n_out_dim = y.shape[1]
            w = np.zeros((n, n_out_dim))
            b = np.zeros(n_out_dim).reshape(1, -1)

            for _ in range(self.iterations):
                y_pred = (xx.dot(w) + b)

                dw = (-(2 * xx.T.dot(y - y_pred)) - self.penalty) / m
                db = - 2 * np.sum((y - y_pred), axis=0).reshape(1, -1) / m

                w = w - self.learning_rate * dw
                b = b - self.learning_rate * db

        return w, b