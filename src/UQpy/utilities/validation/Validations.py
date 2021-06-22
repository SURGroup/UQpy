import functools


def check_copula_theta():
    def validator(f):
        @functools.wraps(f)
        def wrap(self, **kwargs):
            scalar = kwargs['theta']
            if scalar is not None and ((not isinstance(scalar, (float, int))) or
                                       (scalar < -1 or scalar == 0.)):
                raise ValueError('Input theta of Copula {type} should be a float in [-1, +oo).'
                                 .format(type=self.__class__.__name__))
            return f(self, **kwargs)
        return wrap
    return validator

def check_random_state():
    def validator(f):
        @functools.wraps(f)
        def wrap(self, **kwargs):
            random_state = kwargs['random_state']
            if not isinstance(random_state, int) and \
                    not isinstance(random_state, (type(None), np.random.RandomState)):
                raise TypeError('UQpy: random_state must be None, an int or an np.random.RandomState object.')
            return f(self, **kwargs)
        return wrap
    return validator


def check_sample_dimensions():
    def validator(f):
        @functools.wraps(f)
        def wrap(self, **kwargs):
            samples = kwargs['unit_uniform_samples']
            print(type(samples))
            if len(samples.shape) != 2 or samples.shape[1] > 2:
                raise ValueError('Maximum dimension for the {type} Copula is 2'
                                 .format(type=self.__class__.__name__))
            return f(self, **kwargs)
        return wrap
    return validator
