ImportanceSampling
------------------

Importance sampling (IS) is based on the idea of sampling from an alternate distribution and reweighting the samples to be representative of the target distribution (perhaps concentrating sampling in certain regions of the input space that are of greater importance). This often enables efficient evaluations of expectations :math:`E_{ \textbf{x} \sim p} [ f(\textbf{x}) ]` where :math:`f( \textbf{x})` is small outside of a small region of the input space. To this end, a sample :math:`\textbf{x}` is drawn from a proposal distribution :math:`q(\textbf{x})` and re-weighted to correct for the discrepancy between the sampling distribution :math:`q` and the true distribution :math:`p`. The weight of the sample is computed as

.. math:: w(\textbf{x}) = \frac{p(\textbf{x})}{q(\textbf{x})}

If :math:`p` is only known up to a constant, i.e., one can only evaluate :math:`\tilde{p}(\textbf{x})`, where :math:`p(\textbf{x})=\frac{\tilde{p}(\textbf{x})}{Z}`, IS can be used by further normalizing the weights (self-normalized IS). The following figure shows the weighted samples obtained when using IS to estimate a 2D Gaussian target distribution :math:`p`, sampling from a uniform proposal distribution :math:`q`.

.. image:: ../_static/SampleMethods_IS_samples.png
   :scale: 40 %
   :alt: IS weighted samples
   :align: center


ImportanceSampling Class
^^^^^^^^^^^^^^^^^^^^^^^^^

Methods
""""""""""""""""""
.. autoclass:: UQpy.sampling.ImportanceSampling
   :members: run, resample,

Attributes
""""""""""""""""""
.. autoattribute:: UQpy.sampling.ImportanceSampling.samples
.. autoattribute:: UQpy.sampling.ImportanceSampling.unnormalized_log_weights
.. autoattribute:: UQpy.sampling.ImportanceSampling.weights
.. autoattribute:: UQpy.sampling.ImportanceSampling.unweighted_samples
