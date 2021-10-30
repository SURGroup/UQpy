InferenceModel
--------------------------------

For any inference task, the user must first create, for each model studied, an instance of the class ``InferenceModel`` that defines the problem at hand. This class defines an inference model that will serve as input for all remaining inference classes. A model can be defined in various ways. The following summarizes the four types of inference models that are supported by ``UQpy``. These four types are further summarized in the figure below.

* **Case 1a** - `Gaussian error model powered by` ``RunModel``: In this case, the data is assumed to come form a model of the following
  form,  `data ~ h(theta) + eps`, where `eps` is iid Gaussian and `h` consists of a computational
  model executed using ``RunModel``. Data is a 1D ndarray in this setting.
* **Case 1b** - `non-Gaussian error model powered by` ``RunModel``: In this case, the user must provide the likelihood
  function in addition to a ``RunModel`` object. The data type is user-defined and must be consistent with the
  likelihood function definition.
* **Case 2:** - `User-defined likelihood without` ``RunModel``: Here, the likelihood function is user-defined and
  does not leverage ``RunModel``. The data type must be consistent with the likelihood function definition.
* **Case 3:** `Learn parameters of a probability distribution:` Here, the user must define an object of the
  ``Distribution`` class. Data is an ndarray of shape `(ndata, dim)` and consists in `ndata` iid samples from the
  probability distribution.

.. image:: ../_static/Inference_models.png
   :scale: 30 %
   :align: left


Defining a Log-likelihood function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The critical component of the ``InferenceModel`` class is the evaluation of the log-likelihood function. ``InferenceModel`` has been constructed to be flexible in how the user specifies the log-likelihood function. The log-likelihood function can be specified as a user-defined callable method that is passed directly into the ``InferenceModel`` class. As the cases suggest, a user-defined log-likelihood function must take as input, at minimum, both the parameters of the model and the data points at which to evaluate the log-likelihood. It may also take additional keyword arguments. The method may compute the log-likelihood at the data points on its own, or it may rely on a computational model defined through the ``RunModel`` class. If the log-likelihood function relies on a ``RunModel`` object, this object is also passed into ``InferenceModel`` and the log-likelihood method should also take as input, the output (`qoi_list`) of the ``RunModel`` object evaluated at the specified parameter values.

InferenceModel Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: UQpy.inference.inference_models.baseclass.InferenceModel
   :members: