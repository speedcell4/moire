import dynet as dy
import numpy as np

from moire import Expression, uniform

__all__ = [
    'argmax', 'argmin',
    'epsilon_argmax', 'epsilon_argmin',
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


if __name__ == '__main__':
    x = dy.inputVector([1, 2, 3, 4])
    print(np.histogram([epsilon_argmax(x, 0.5) for _ in range(10000)], bins=4))
