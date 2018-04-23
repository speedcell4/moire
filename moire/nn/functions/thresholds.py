import dynet as dy
import numpy as np

import moire
from moire import Expression
from moire.nn.modules import Function


def linear(x: Expression) -> Expression:
    return x


relu = dy.rectify


class LeakyRelu(Function):
    def __init__(self, alpha: float = 0.2) -> None:
        super(LeakyRelu, self).__init__()
        self.alpha = -alpha

    @property
    def gain(self) -> float:
        return float(np.sqrt(2.0 / (1.0 + self.alpha ** 2)))

    def __call__(self, x: Expression) -> Expression:
        return relu(x) + relu(-x) * self.alpha


if __name__ == '__main__':
    leaky_relu = LeakyRelu()
    x = dy.inputVector([-2, -1, 0, 1, 2])
    moire.debug(leaky_relu(x).value())
    moire.debug(leaky_relu.gain)
