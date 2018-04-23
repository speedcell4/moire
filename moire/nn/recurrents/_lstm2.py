import dynet as dy
import numpy as np

import moire
from moire import Expression, ParameterCollection, nn
from moire.nn.initializers import ConcatenatedInitializer, GlorotNormal, One, Orthogonal, Uniform, Zero
from moire.nn.sigmoids import sigmoid
from moire.nn.trigonometry import tanh


class LSTM(nn.Module):
    def __init__(self, pc: ParameterCollection, input_size: int, hidden_size: int,
                 dropout_ratio: float = None, zoneout_ratio: float = None,
                 activation=tanh, recurrent_activation=sigmoid,
                 kernel_initializer=GlorotNormal(), recurrent_initializer=Orthogonal(),
                 hidden_initializer=Uniform(), bias_initializer=Zero(), forget_bias_initializer=One()):
        super().__init__(pc)

        self.U = self.add
