import dynet as dy

from moire import nn, Expression


def argmax(x: Expression, axis: int = None) -> int:
    return int(x.npvalue().argmax(axis=axis))


def argmin(x: Expression, axis: int = None) -> int:
    return int(x.npvalue().argmin(axis=axis))


class EpsilonArgmax(nn.Function):
    def __init__(self, epsilon: float, axis: int = None):
        super(EpsilonArgmax, self).__init__()
        self.epsilon = epsilon
        self.axis = axis

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} : {self.epsilon}>'

    def __call__(self, x: Expression) -> int:
        raise NotImplementedError


class EpsilonArgmin(nn.Function):
    def __init__(self, epsilon: float, axis: int = None):
        super(EpsilonArgmin, self).__init__()
        self.epsilon = epsilon
        self.axis = axis

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} : {self.epsilon}>'

    def __call__(self, x: Expression) -> int:
        raise NotImplementedError


if __name__ == '__main__':
    x = dy.inputVector([1, 2, 3])
    print(argmax(x))
