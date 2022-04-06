"""

Nataf
==============================================

"""

# %% md
#
# We'll be using UQpy's Nataf transformation functionalities. We also use Matplotlib to display results graphically.
#
# Additionally, this demonstration opts to use Numpy's random state management to ensure that results are reproducible
# between notebook runs.

# %%

from UQpy.transformations import Nataf
import numpy as np
import matplotlib.pyplot as plt
from UQpy.distributions import Normal, Gamma, Lognormal
from UQpy.transformations import Decorrelate, Correlate


# %% md
#
# We will start by constructing two non-Gaussian random variables. The first one, follows a :code:`Gamma` distribution
# :code:`dist1`,  while the second one follows a :code:`Lognormal` distribution  :code:`dist2`.
# The two distributions are correlated according to the correlation matrix :code:`Rx`.

# %%

dist1 = Gamma(4.0, loc=0.0, scale=1.0)
dist2 = Lognormal(s=2., loc=0., scale=np.exp(1))
Rx = np.array([[1.0, 0.9], [0.9, 1.0]])

# %% md
#
# Next, we'll construct a :code:`Nataf` object :code:`nataf_obj`. Here, we provide the distribution of the random
# variables :code:`distributions` and the correlation matrix :code:`corr_x` in the parameter space.

# %%

nataf_obj = Nataf(distributions=[dist1, dist2], corr_x=Rx)

# %% md
#
# We can use the :code:`rvs` method of the :code:`nataf_obj` object to draw random samples from the two distributions.

# %%

samples_x = nataf_obj.rvs(1000)

# %% md
#
# We can visualize the samples by plotting them on axes of each distribution's range.

# %%


plt.figure()
plt.title('non-Gaussian random variables')
plt.scatter(samples_x[:, 0], samples_x[:, 1])
plt.grid(True)
plt.xlabel('$X_1$')
plt.ylabel('$X_2$')
plt.show()

# %% md
#
# We can invoke the :code:`run` method of the :code:`nataf_obj` object to transform the non-Gaussian random variables
# from space :code:`X`to the correlated standard normal space :code:`Z`. Here, we provide as :code:`boolean (True)` an
# optional attribute :code:`jacobian` to calculate the Jacobian of the transformation. The distorted correlation matrix
# can be accessed in the attribute :code:`corr_z`

# %%

nataf_obj.run(samples_x=samples_x, jacobian=True)
print(nataf_obj.corr_z)


# %% md
#
# We can visualize the correlated (standard normal) samples by plotting them on axes of each distribution's range.

# %%


plt.figure()
plt.title('Correlated standard normal samples')
plt.scatter(nataf_obj.samples_z[:, 0], nataf_obj.samples_z[:, 1])
plt.grid(True)
plt.xlabel('$Z_1$')
plt.ylabel('$Z_2$')
plt.show()

# %% md
#
# We can use the :code:`Decorrelate` class to transform the correlated standard normal samples to the uncorrelated
# standard normal space :code:`U`.

# %%

samples_u = Decorrelate(nataf_obj.samples_z, nataf_obj.corr_z).samples_u

# %% md
#
# We can visualize the uncorrelated (standard normal) samples by plotting them on axes of each distribution's range.

# %%

plt.figure()
plt.title('Uncorrelated standard normal samples')
plt.scatter(samples_u[:, 0], samples_u[:, 1])
plt.grid(True)
plt.xlabel('$U_1$')
plt.ylabel('$U_2$')
plt.show()

# %% md
#
# We can use the :code:`Correlate` class to transform the uncorrelated standard normal samples back to the correlated
# standard normal space :code:`Z`.

# %%

samples_z = Correlate(samples_u, nataf_obj.corr_z).samples_z

# %% md
#
# We can visualize the correlated (standard normal) samples by plotting them on axes of each distribution's range.

# %%

plt.figure()
plt.title('Correlated standard normal samples')
plt.scatter(samples_z[:, 0], samples_z[:, 1])
plt.grid(True)
plt.xlabel('$U_1$')
plt.ylabel('$U_2$')
plt.show()


# %% md
#
# In the second example, we will calculate the distortion of the correlation coefficient (in the standard normal space)
# as a function of the correlation coefficient (in the parameter space).  To this end, we consider N=20 values for the
# latter, ranging from -0.999 to 0.999. We do not consider the values -1 and 1 since they result in numerical
# instabilities.

# %%

N = 20
w3 = np.zeros(N)
rho = np.linspace(-0.999, 0.999, N)
for i in range(N):
    Rho1 = np.array([[1.0, rho[i]], [rho[i], 1.0]])
    ww = Nataf([dist1, dist2], corr_x=Rho1).corr_z
    w3[i] = ww[0, 1]

plt.plot(rho, w3)
plt.xlabel('rho_X')
plt.ylabel('rho_Z')
plt.show()


# %% md
#
# In the third example, we will calculate the distortion of the correlation coefficient (in the parameter space) as a
# function of the correlation coefficient (in the standard normal space).  To this end, we consider N=20 values for the
# latter, ranging from -0.999 to 0.999. We do not consider the values -1 and 1 since they result in numerical
# instabilities.

# %%

w4 = np.zeros(N)
rho = np.linspace(-0.999, 0.999, N)
for i in range(N):
    Rho1 = np.array([[1.0, rho[i]], [rho[i], 1.0]])
    ww = Nataf(distributions=[dist1, dist2], corr_z=Rho1).corr_x
    w4[i] = ww[0, 1]

plt.plot(rho, w4)
plt.plot(rho, rho)
plt.show()
