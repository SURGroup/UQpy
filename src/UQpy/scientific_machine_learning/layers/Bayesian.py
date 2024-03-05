from UQpy.scientific_machine_learning.baseclass.Layer import Layer


class Bayesian(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        ...