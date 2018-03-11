import moire
from moire import nn, Expression, bernoulli, uniform

__all__ = [
    'argmax', 'argmin',
    'EpsilonArgMax', 'EpsilonArgMin',
]


def argmax(x: Expression, axis: int = None) -> int:
    return int(x.npvalue().argmax(axis=axis))


def argmin(x: Expression, axis: int = None) -> int:
    return int(x.npvalue().argmin(axis=axis))


class EpsilonArgMax(nn.Function):
    def __init__(self, epsilon: float, axis: int = None):
        super(EpsilonArgMax, self).__init__()
        self.epsilon = epsilon
        self.axis = axis

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} : {self.epsilon}>'

    def __call__(self, x: Expression) -> int:
        if moire.config.train and bernoulli(p=self.epsilon).value():
            dim, batch_size = x.dim()
            return int(uniform(low=0, high=dim[0]).value())
        return argmax(x, self.axis)


class EpsilonArgMin(nn.Function):
    def __init__(self, epsilon: float, axis: int = None):
        super(EpsilonArgMin, self).__init__()
        self.epsilon = epsilon
        self.axis = axis

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} : {self.epsilon}>'

    def __call__(self, x: Expression) -> int:
        if moire.config.train and bernoulli(p=self.epsilon).value():
            dim, batch_size = x.dim()
            return int(uniform(low=0, high=dim[0]).value())
        return argmin(x, self.axis)
