import dynet as dy

import moire
from moire import nn, where, bernoulli_like

__all__ = [
    'Dropout', 'Zoneout',
]


class Dropout(nn.Function):
    __slots__ = ('ratio',)

    def __init__(self, ratio: float = None):
        super(Dropout, self).__init__()
        self.ratio = ratio

    def __call__(self, x):
        if self.ratio is not None and moire.config.train:
            return dy.dropout(x, self.ratio)
        return x


class Zoneout(nn.Function):
    __slots__ = ('ratio',)

    def __init__(self, ratio: float = None):
        super(Zoneout, self).__init__()
        self.ratio = None if ratio is None else 1.0 - ratio

    def __call__(self, x, h):
        if self.ratio is not None and moire.config.train:
            cond = bernoulli_like(x, self.ratio)
            return where(cond, x, h)
        return x


if __name__ == '__main__':
    zoneout = Zoneout()

    x = dy.inputVector([1, 2, 3])
    h = dy.inputVector([7, 8, 9])

    print(zoneout(x, h).value())
