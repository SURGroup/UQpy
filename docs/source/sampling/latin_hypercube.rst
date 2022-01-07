Latin Hypercube Sampling
------------------------

The :class:`.LatinHypercubeSampling` class generates random samples from a specified probability distribution(s) using
Latin hypercube sampling. LatinHypercubeSampling has the advantage that the samples generated are uniformly distributed
over each marginal distribution. LatinHypercubeSampling is performed by dividing the range of each random variable
into :math:`N` bins with equal probability mass, where :math:`N` is the required number of samples, generating one
sample per bin, and then randomly pairing the samples.


.. toctree::
   :hidden:
   :maxdepth: 1

    Latin Hypercube Class <latin_hypercube/lhs_class>
    List of Available Latin Hypercube Criteria <latin_hypercube/lhs_criteria>
    Adding new Latin Hypercube Design Criteria <latin_hypercube/lhs_user_criterion>
    Examples <../auto_examples/sampling/latin_hypercube/index>





