"""
Inverse FORM - Cantilever Beam
-----

A cantilever beam (example 7.2 in :cite:`FORM_XDu`) is considered to fail if the displacement at the tip exceeds the
threshold :math:`D_0`. The performance function :math:`G(\textbf{U})` of this problem is given by

.. math:: G = D_0 - \frac{4L^3}{Ewt} \sqrt{ \left(\frac{P_x}{w^2}\right)^2 + \left(\frac{P_y}{t^2}\right)^2}

Where the external forces are modeled as random variables :math:`P_x \sim N(500, 100)` and :math:`P_y \sim N(1000,100)`.
The constants in the problem are length (:math:`L=100`), elastic modulus (:math:E=30\times 10^6), cross section width
(:math:`w=2`) and cross section height (:math:`t=4`).

"""
# %% md
#
# Import the necessary modules.

# %%
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from scipy import stats
from UQpy.distributions import Normal
from UQpy.reliability.taylor_series import InverseFORM
from UQpy.run_model.RunModel import RunModel
from UQpy.run_model.model_execution.PythonModel import PythonModel

# %% md
#
# Next, we initialize the :code:`RunModel` object.
# The file defining the performance function file can be found on the UQpy GitHub.
# It contains a function :code:`cantilever_beam` to compute the performance function :math:`G(\textbf{U})`.

# %%

model = PythonModel(model_script='performance_function.py', model_object_name="cantilever_beam")
runmodel_object = RunModel(model=model)

# %% md
#
# Next, we define the external forces in the :math:`x` and :math:`y` direction as distributions that will be passed into
# :code:`FORM`. Along with the distributions, :code:`FORM` takes in the previously defined :code:`runmodel_object`,
# the specified probability of failure, and the tolerances. These tolerances are smaller than the defaults to ensure
# convergence with the level of accuracy given in the problem.

# %%

p_fail = 0.04054
distributions = [Normal(500, 100), Normal(1_000, 100)]
inverse_form = InverseFORM(distributions=distributions,
                           runmodel_object=runmodel_object,
                           p_fail=p_fail,
                           tolerance_u=1e-5,
                           tolerance_gradient=1e-5)

# %% md
#
# With everything defined we are ready to run the inverse first-order reliability method and print the results.
# The solution to this problem given by Du is :math:`\textbf{U}^*=(1.7367, 0.16376)` with a reliability index of
# :math:`\beta_{HL}=||\textbf{U}^*||=1.7444` and probability of failure of
# :math:`p_{fail} = \Phi(-\beta_{HL})=\Phi(-1.7444)=0.04054`. We expect this problem to converge in 4 iterations.
# We confirm our design point matches this length, and therefore has a probability of failure specified by our input.

# %%

inverse_form.run()
beta = np.linalg.norm(inverse_form.design_point_u)
print('Design point in standard normal space (u^*):', inverse_form.design_point_u[0])
print('Design point in original space:', inverse_form.design_point_x[0])
print('Hasofer-Lind reliability index:', beta)
print('Probability of failure at design point:', stats.norm.cdf(-beta))
print('Number of iterations:', inverse_form.iteration_record[0])