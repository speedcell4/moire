import dynet as dy
import numpy as np

from moire import Expression, uniform

__all__ = [
    'argmax', 'argmin',
    'epsilon_argmax', 'epsilon_argmin',
    'gumbel_argmax', 'gumbel_argmin',
]


def argmax(x: Expression, axis: int = None) -> int:
    return int(x.npvalue().argmax(axis=axis))


def argmin(x: Expression, axis: int = None) -> int:
    return int(x.npvalue().argmin(axis=axis))


def epsilon_argmax(x: Expression, epsilon: float, axis: int = None) -> int:
    if np.random.uniform(low=0.0, high=1.0, size=()) < epsilon:
        dim, batch_size = x.dim()
        return int(uniform(low=0, high=dim[0]).value())
    return argmax(x, axis)


def epsilon_argmin(x: Expression, epsilon: float, axis: int = None) -> int:
    if np.random.uniform(low=0.0, high=1.0, size=()) < epsilon:
        dim, batch_size = x.dim()
        return int(uniform(low=0, high=dim[0]).value())
    return argmin(x, axis)


def gumbel_argmax(prob: Expression, loc: float = 0.0, scale: float = 1.0, axis: int = None) -> int:
    shape, batch_size = prob.dim()
    a = dy.inputVector(np.random.gumbel(loc=loc, scale=scale, size=shape))
    return int(np.argmax((dy.log(prob) + a).value(), axis=axis).astype(np.int32, copy=False))


def gumbel_argmin(prob: Expression, loc: float = 0.0, scale: float = 1.0, axis: int = None) -> int:
    shape, batch_size = prob.dim()
    a = dy.inputVector(np.random.gumbel(loc=loc, scale=scale, size=shape))
    return int(np.argmin((dy.log(prob) + a).value(), axis=axis).astype(np.int32, copy=False))


if __name__ == '__main__':
    x = dy.inputVector([1, 2, 3, 4])
    a = np.array([gumbel_argmin(dy.softmax(x)) for _ in range(1000)])
    print(np.histogram(a, bins=4))
