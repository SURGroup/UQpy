from UQpy.scientific_machine_learning.baseclass.Layer import Layer
# Modified by George and Ponkrshnan
import torch.nn as nn

class Dropout(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, p = 0.5):
        self.drop_rate = p
        # this should be determined based on the dimensions of x N x C x H x W
        # N x features
        if len(x.shape) > 2:
            self.dropout = nn.Dropout2d(self.drop_rate)
        else:
            self.dropout = nn.Dropout(self.drop_rate)
        x = self.dropout(x)
