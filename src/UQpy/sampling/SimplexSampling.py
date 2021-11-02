import scipy.stats as stats
from beartype import beartype
from UQpy.utilities.ValidationTypes import *
from UQpy.utilities.Utilities import process_random_state


class SimplexSampling:
    @beartype
    def __init__(
        self,
        nodes=None,
        samples_number: PositiveInteger = None,
        random_state: RandomStateType = None,
    ):
        """
        Generate uniform random samples inside an n-dimensional simplex.

        :param nodes: The vertices of the simplex.
        :param samples_number: The number of samples to be generated inside the simplex.
         If `samples_number` is provided when the object is defined, the :meth:`run` method will be called
         automatically. If `samples_number` is not provided when the object is defined, the user must invoke the
         :meth:`run` method and specify `samples_number`.
        :param random_state: Random seed used to initialize the pseudo-random number generator. Default is None.
         If an integer is provided, this sets the seed for an object of :class:`numpy.random.RandomState`. Otherwise,
         the object itself can be passed directly.
        """
        self.nodes = np.atleast_2d(nodes)
        self.samples_number = samples_number

        if self.nodes.shape[0] != self.nodes.shape[1] + 1:
            raise NotImplementedError(
                "UQpy: Size of simplex (nodes) is not consistent."
            )

        self.random_state = process_random_state(random_state)

        if samples_number is not None:
            self.samples = self.run(samples_number=samples_number)
            """New random samples distributed uniformly inside the simplex."""

    @beartype
    def run(self, samples_number: PositiveInteger):
        """
        Execute the random sampling in the :class:`.SimplexSampling` class.
        The :meth:`run` method is the function that performs random sampling in the :class:`.SimplexSampling`` class.
        If `samples_number` is provided called when the :class:`.SimplexSampling` object is defined, the
        :meth:`run` method is automatically. The user may also call the :meth:`run` method directly to generate samples.
        The :meth:`run` method of the :class:`.SimplexSampling` class can be
        invoked many times and each time the generated samples are appended to the existing samples.

        :param samples_number: Number of samples to be generated inside the simplex.
         If the :meth:`run` method is invoked multiple times, the newly generated samples will be appended to the
         existing samples.
        :return: The :meth:`run` method has no returns, although it creates and/or appends the `samples` attribute of the
         :class:`.SimplexSampling` class.
        """
        self.samples_number = samples_number
        dimension = self.nodes.shape[1]
        if dimension > 1:
            sample = np.zeros([self.samples_number, dimension])
            for i in range(self.samples_number):
                r = np.zeros([dimension])
                ad = np.zeros(shape=(dimension, len(self.nodes)))
                for j in range(dimension):
                    b_ = list()
                    for k in range(1, len(self.nodes)):
                        ai = self.nodes[k, j] - self.nodes[k - 1, j]
                        b_.append(ai)
                    ad[j] = np.hstack((self.nodes[0, j], b_))
                    r[j] = stats.uniform.rvs(
                        loc=0, scale=1, random_state=self.random_state
                    ) ** (1 / (dimension - j))
                d = np.cumprod(r)
                r_ = np.hstack((1, d))
                sample[i, :] = np.dot(ad, r_)
        else:
            a = min(self.nodes)
            b = max(self.nodes)
            sample = a + (b - a) * stats.uniform.rvs(
                size=[self.samples_number, dimension], random_state=self.random_state
            )
        return sample

    def __copy__(self):
        new = self.__class__(nodes=self.nodes, random_state=self.random_state)
        new.__dict__.update(self.__dict__)

        return new
