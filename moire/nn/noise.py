import dynet as dy

from moire import nn, where, bernoulli_like

__all__ = [
    'Dropout', 'Zoneout',
]


class Dropout(nn.Function):
    def __init__(self, ratio: float):
        super(Dropout, self).__init__()
        self.ratio = ratio

    def __call__(self, x):
        if self.training and self.ratio > 0.0:
            return dy.dropout(x, self.ratio)
        return x


class Zoneout(nn.Function):
    def __init__(self, ratio: float):
        super(Zoneout, self).__init__()
        self.ratio = 1.0 - ratio

    def __call__(self, x, h):
        if self.training and self.ratio < 1.0:
            cond = bernoulli_like(x, self.ratio)
            return where(cond, x, h)
        return x


if __name__ == '__main__':
    zoneout = Zoneout(0.9)
    zoneout.training = True

    x = dy.inputVector([1, 2, 3])
    h = dy.inputVector([7, 8, 9])

    print(zoneout(x, h).value())
