import numpy as np

from UQpy.SampleMethods.Strata.strata import Strata


class RectangularStrata(Strata):
    """
    Define a geometric decomposition of the n-dimensional unit hypercube into disjoint and space-filling
    rectangular strata.

    ``RectangularStrata`` is a child class of the ``Strata`` class

    **Inputs:**

    * **nstrata** (`list` of `int`):
        A list of length `n` defining the number of strata in each of the `n` dimensions. Creates an equal
        stratification with strata widths equal to 1/`n_strata`. The total number of strata, `N`, is the product
        of the terms of `n_strata`.

        Example: `n_strata` = [2, 3, 2] creates a 3-dimensional stratification with:\n
                2 strata in dimension 0 with stratum widths 1/2\n
                3 strata in dimension 1 with stratum widths 1/3\n
                2 strata in dimension 2 with stratum widths 1/2\n

        The user must pass one of `nstrata` OR `input_file` OR `seeds` and `widths`

    * **input_file** (`str`):
        File path to an input file specifying stratum seeds and stratum widths.

        This is typically used to define irregular stratified designs.

        The user must pass one of `n_strata` OR `input_file` OR `seeds` and `widths`

    * **seeds** (`ndarray`):
        An array of dimension `N x n` specifying the seeds of all strata. The seeds of the strata are the
        coordinates of the stratum orthotope nearest the global origin.

        Example: A 2-dimensional stratification with 2 equal strata in each dimension:

            `origins` = [[0, 0], [0, 0.5], [0.5, 0], [0.5, 0.5]]

        The user must pass one of `n_strata` OR `input_file` OR `seeds` and `widths`

    * **widths** (`ndarray`):
        An array of dimension `N x n` specifying the widths of all strata in each dimension

        Example: A 2-dimensional stratification with 2 strata in each dimension

            `widths` = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]

        The user must pass one of `n_strata` OR `input_file` OR `seeds` and `widths`

    * **random_state** (None or `int` or ``numpy.random.RandomState`` object):
        Random seed used to initialize the pseudo-random number generator. Default is None.

        If an integer is provided, this sets the seed for an object of ``numpy.random.RandomState``. Otherwise, the
        object itself can be passed directly.

    * **verbose** (`Boolean`):
        A boolean declaring whether to write text to the terminal.


    **Attributes:**

    * **nstrata** (`list` of `int`):
        A list of length `n` defining the number of strata in each of the `n` dimensions. Creates an equal
        stratification with strata widths equal to 1/`n_strata`. The total number of strata, `N`, is the product
        of the terms of `n_strata`.

    * **seeds** (`ndarray`):
        An array of dimension `N x n` specifying the seeds of all strata. The seeds of the strata are the
        coordinates of the stratum orthotope nearest the global origin.

    * **widths** (`ndarray`):
        An array of dimension `N x n` specifying the widths of all strata in each dimension

    * **volume** (`ndarray`):
        An array of dimension `(nstrata, )` containing the volume of each stratum. Stratum volumes are equal to the
        product of the strata widths.

    **Methods:**
    """
    def __init__(self, nstrata=None, input_file=None, seeds=None, widths=None, random_state=None, verbose=False):
        super().__init__(random_state=random_state, seeds=seeds, verbose=verbose)

        self.input_file = input_file
        self.nstrata = nstrata
        self.widths = widths

        self.stratify()

    def stratify(self):
        """
        Performs the rectangular stratification.
        """
        if self.verbose:
            print('UQpy: Creating Rectangular stratification ...')

        if self.nstrata is None:
            if self.input_file is None:
                if self.widths is None or self.seeds is None:
                    raise RuntimeError('UQpy: The strata are not fully defined. Must provide `n_strata`, `input_file`, '
                                       'or `seeds` and `widths`.')

            else:
                # Read the strata from the specified input file
                # See documentation for input file formatting
                array_tmp = np.loadtxt(self.input_file)
                self.seeds = array_tmp[:, 0:array_tmp.shape[1] // 2]
                self.widths = array_tmp[:, array_tmp.shape[1] // 2:]

                # Check to see that the strata are space-filling
                space_fill = np.sum(np.prod(self.widths, 1))
                if 1 - space_fill > 1e-5:
                    raise RuntimeError('UQpy: The stratum design is not space-filling.')
                if 1 - space_fill < -1e-5:
                    raise RuntimeError('UQpy: The stratum design is over-filling.')

        # Define a rectilinear stratification by specifying the number of strata in each dimension via nstrata
        else:
            self.seeds = np.divide(self.fullfact(self.nstrata), self.nstrata)
            self.widths = np.divide(np.ones(self.seeds.shape), self.nstrata)

        self.volume = np.prod(self.widths, axis=1)

        if self.verbose:
            print('UQpy: Rectangular stratification created.')

    @staticmethod
    def fullfact(levels):

        """
        Create a full-factorial design

        Note: This function has been modified from pyDOE, released under BSD License (3-Clause)\n
        Copyright (C) 2012 - 2013 - Michael Baudin\n
        Copyright (C) 2012 - Maria Christopoulou\n
        Copyright (C) 2010 - 2011 - INRIA - Michael Baudin\n
        Copyright (C) 2009 - Yann Collette\n
        Copyright (C) 2009 - CEA - Jean-Marc Martinez\n
        Original source code can be found at:\n
        https://pythonhosted.org/pyDOE/#\n
        or\n
        https://pypi.org/project/pyDOE/\n
        or\n
        https://github.com/tisimst/pyDOE/\n

        **Input:**

        * **levels** (`list`):
            A list of integers that indicate the number of levels of each input design factor.

        **Output:**

        * **ff** (`ndarray`):
            Full-factorial design matrix.
        """
        # Number of factors
        n_factors = len(levels)
        # Number of combinations
        n_comb = np.prod(levels)
        ff = np.zeros((n_comb, n_factors))

        level_repeat = 1
        range_repeat = np.prod(levels)
        for i in range(n_factors):
            range_repeat //= levels[i]
            lvl = []
            for j in range(levels[i]):
                lvl += [j] * level_repeat
            rng = lvl * range_repeat
            level_repeat *= levels[i]
            ff[:, i] = rng

        return ff

    def plot_2d(self):
        """
        Plot the rectangular stratification.

        This is an instance method of the ``RectangularStrata`` class that can be called to plot the boundaries of a
        two-dimensional ``RectangularStrata`` object on :math:`[0, 1]^2`.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig = plt.figure()
        ax = fig.gca()
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        for i in range(self.seeds.shape[0]):
            rect1 = patches.Rectangle(self.seeds[i, :], self.widths[i, 0], self.widths[i, 1], linewidth=1,
                                      edgecolor='b', facecolor='none')
            ax.add_patch(rect1)

        return fig

