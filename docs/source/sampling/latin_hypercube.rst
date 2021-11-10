Latin Hypercube Sampling
------------------------

The :class:`.LatinHypercubeSampling` class generates random samples from a specified probability distribution(s) using Latin hypercube sampling. LatinHypercubeSampling has the advantage that the samples generated are uniformly distributed over each marginal distribution. LatinHypercubeSampling is perfomed by dividing the range of each random variable into N bins with equal probability mass, where N is the required number of samples, generating one sample per bin, and then randomly pairing the samples.


LatinHypercubeSampling Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: UQpy.sampling.LatinHypercubeSampling
    :members:
    :private-members:

Adding New Latin Hypercube Design Criteria
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.LatinHypercubeSampling` class offers a variety of methods for pairing the samples in a Latin hypercube
design. These are specified by the `criterion` parameter (i.e. Random, Centered, MaxiMin, MinCorrelation).
Each one of the Criteria classes can be found in :py:mod:`.latin_hypercube_criteria` folder, with the :class:`.Criterion` baseclass
defining their common interface. As a result, adding a new method is straightforward.
This is done by creating a new class that implements the :class:`.Criterion` abstract baseclass. The base class requires the
creation of a :meth:`generate_samples` method that contains the algorithm for pairing the samples.
This method retrieves the randomly generated samples generated in the baseclass in equal probability bins in each
dimension and returns a set of samples that is paired according to the user's desired criterion.
The user may also pass criterion-specific parameters into the :meth:`__init__` method of the generated class.
The output of this function should be a numpy array of at least two-dimensions with the first dimension being the
number of samples and the second dimension being the number of variables.

.. autoclass:: UQpy.sampling.Criterion
    :members:

An example of a user-defined criterion is given below:


>>> class UserCriterion(Criterion):
>>>
>>>     def __init__(self, random_state: RandomStateType = None):
>>>         super().__init__(random_state)
>>>
>>>     def generate_samples(self):
>>>         lhs_samples = np.zeros_like(self.samples)
>>>         samples_number = len(self.samples)
>>>         for j in range(self.samples.shape[1]):
>>>             if self.random_state is not None:
>>>                 order = self.random_state.permutation(samples_number)
>>>             else:
>>>                 order = np.random.permutation(samples_number)
>>>             lhs_samples[:, j] = self.samples[order, j]
>>>
>>>         return lhs_samples

