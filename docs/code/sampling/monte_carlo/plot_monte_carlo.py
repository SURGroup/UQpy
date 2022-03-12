"""

Monte Carlo sampling with UQpy
==================================
"""

# %% md
#
# We'll be using UQpy's Monte Carlo sampling functionalities. We also use Matplotlib to display results graphically.
#
# Additionally, this demonstration opts to use Numpy's random state management to ensure that results are reproducible
# between notebook runs.

# %%

from UQpy.sampling import MonteCarloSampling
import matplotlib.pyplot as plt

from numpy.random import RandomState

# %% md
#
# Step-by-step: continuous univariate distribution
# -------------------------------------------------
# First, we import UQpy's normal distribution class.

# %%
from UQpy.distributions import Normal

# %% md
#
# We'll start by constructing two identical standard normal distributions :code:`normal1` and :code:`normal2`

# %%

normal1 = normal2 = Normal()

# %% md
#
# Next, we'll construct a :code:`MonteCarloSampling` object :code:`mc` to generate random samples following those
# distributions. Here, we specify an optional initial number of samples, :code:`nsamples` to be generated at the
# object's construction. For teh purposes of this demonstration, we also supply a random seed :code:`random_state`.
#
# We access the generated samples via the :code:`samples` attribute.

# %%

mc = MonteCarloSampling(distributions=[normal1, normal2],
                        nsamples=5,
                        random_state=RandomState(123))

mc.samples

# %% md
#
# To generate more samples on :code:`mc` after construction, we call :code:`mc.run` and once again specify
# :code:`nsamples`.

# %%

mc.run(nsamples=2, random_state=RandomState(23))

mc.samples

# %% md
#
# We can transform the samples onto the unit hypercube via applying the probability integral transformation on the
# samples to yield similar samples from the uniform distribution. We call :code:`mc.transform_u01`, from which results
# are stored in the :code:`samplesU01` attribute.

# %%

mc.transform_u01()

mc.samplesU01

# %% md
#
# We can visualize the (untransformed) samples by plotting them on axes of each distribution's range.

# %%

fig, ax = plt.subplots()
plt.title('Samples')

plt.scatter(x=mc.samples[:, 0],
            y=mc.samples[:, 1],
            marker='o')

plt.setp(ax, xlim=(-1.7, 1.7), ylim=(-2.6, 2.6))
ax.yaxis.grid(True)
ax.xaxis.grid(True)

# %% md
#
# As well, we can visualize each distribution's sample densities via histograms.

# %%

fig, ax = plt.subplots(1, 2)

for i in (0, 1):
    ax[i].set_title('Distribution ' + str(i + 1))
    ax[i].hist(mc.samples[:, i])
    ax[i].yaxis.grid(True)
    ax[i].xaxis.grid(True)

plt.setp(ax, xlim=(-3, 3), ylim=(0, 2));

# %% md
#
# Additional Examples
# -------------------------------------------------
# Continuous multivariate distribution
# """"""""""""""""""""""""""""""""""""""""
# We'll use UQpy's multivariate normal class.

# %%

from UQpy.distributions import MultivariateNormal

# %% md
#
# And we construct a multivariate normal distribution :code:`mvnormal` specifying parameters :code:`mean` with a vector
# of mean values and :code:`cov` with a covariance matrix.

# %%

mvnormal = MultivariateNormal(mean=[1, 2],
                              cov=[[4, -0.9],
                                   [-0.9, 1]])

# %% md
#
# With this distribution, we construct a :code:`MonteCarloSampling` object :code:`mvmc` and generate five samples on
# construction.

# %%

mvmc = MonteCarloSampling(distributions=mvnormal,
                          nsamples=5,
                          random_state=RandomState(456))

mvmc.samples

# %% md
#
# Mixing a multivariate and a univariate continuous distribution
# """"""""""""""""""""""""""""""""""""""""
# Here, we use one of our normal distributions and our multivariate normal distribution. Notice how each distribution
# has its own bundle (array) of samples per run of sampling—even when that bundle contains a single value.

# %%

mixedmc = MonteCarloSampling(distributions=[normal1, mvnormal],
                             nsamples=5,
                             random_state=RandomState(789))

mixedmc.samples

# %% md
#
# Mixing a continuous and a discrete distribution
# """""""""""""""""""""""""""""""""""""""""""""""""
# We'll use UQpy's binomial distribution class for our discrete distribution.

# %%


from UQpy.distributions import Binomial

# %% md
#
# With that, we'll construct a :code:`binomial` distribution binomial with five trials :code:`n` and a 40% probability
# :code:`p` of success per trial.

# %%

binomial = Binomial(n=5, p=0.4)

# %% md
#
# And we construct a :code:`MonteCarloSampling` object :code:`cdmv` with five initial samples using our binomial
# distribution and one of our normal distributions.

# %%

cdmv = MonteCarloSampling(distributions=[binomial, normal1],
                          nsamples=5,
                          random_state=RandomState(333))

cdmv.samples
