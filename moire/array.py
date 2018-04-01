import dynet as dy
import numpy as np

import moire
from moire import Expression

__all__ = [
    'zeros', 'ones', 'full', 'normal', 'bernoulli', 'uniform', 'gumbel',
    'zeros_like', 'ones_like', 'full_like', 'normal_like', 'bernoulli_like', 'uniform_like', 'gumbel_like',
    'eye', 'diagonal',
    'where',
]


def zeros(*dim, batch_size: int = 1) -> Expression:
    a = np.zeros((*dim, batch_size), dtype=np.float32)
    return dy.inputTensor(a, batched=True, device=moire.config.device)


def zeros_like(x: Expression) -> Expression:
    dim, batch_size = x.dim()
    return zeros(*dim, batch_size=batch_size)


def ones(*dim, batch_size: int = 1) -> Expression:
    a = np.ones((*dim, batch_size), dtype=np.float32)
    return dy.inputTensor(a, batched=True, device=moire.config.device)


def ones_like(x: Expression) -> Expression:
    dim, batch_size = x.dim()
    return ones(*dim, batch_size=batch_size)


def eye(N: int, M: int = None, k: int = 0) -> Expression:
    return dy.inputTensor(np.eye(N, M, k))


def diagonal(x: Expression) -> Expression:
    (dim0, dim1), batch_size = x.dim()
    return dy.cmult(x, eye(dim0, dim1))


def full(*dim, value, batch_size: int = 1) -> Expression:
    a = np.full((*dim, batch_size), fill_value=value, dtype=np.float32)
    return dy.inputTensor(a, batched=True, device=moire.config.device)


def full_like(x: Expression, value) -> Expression:
    dim, batch_size = x.dim()
    return full(*dim, value=value, batch_size=batch_size)


def normal(*dim, mean: float = 0.0, stddev: float = 1.0, batch_size: int = 1) -> Expression:
    a = np.random.normal(loc=mean, scale=stddev, size=(*dim, batch_size)).astype(np.float32)
    return dy.inputTensor(a, batched=True, device=moire.config.device)


def normal_like(x: Expression, mean: float = 0.0, stddev: float = 1.0) -> Expression:
    dim, batch_size = x.dim()
    return normal(*dim, mean=mean, stddev=stddev, batch_size=batch_size)


def bernoulli(*dim, p: float, batch_size: int = 1) -> Expression:
    a = np.random.uniform(low=0, high=1.0, size=(*dim, batch_size)) < p
    return dy.inputTensor(a.astype(np.int32), batched=True, device=moire.config.device)


def bernoulli_like(x: Expression, p: float) -> Expression:
    dim, batch_size = x.dim()
    return bernoulli(*dim, p=p, batch_size=batch_size)


def uniform(*dim, low: float, high: float, batch_size: int = 1) -> Expression:
    a = np.random.uniform(low=low, high=high, size=(*dim, batch_size))
    return dy.inputTensor(a, batched=True, device=moire.config.device)


def uniform_like(x: Expression, low: float, high: float) -> Expression:
    dim, batch_size = x.dim()
    return uniform(dim, low=low, high=high, batch_size=batch_size)


def gumbel(*dim, mu: float = 0.0, beta: float = 1.0, batch_size: int = 1) -> Expression:
    a = np.random.gumbel(loc=mu, scale=beta, size=(*dim, batch_size))
    return dy.inputTensor(a, batched=True, device=moire.config.device)


def gumbel_like(x: Expression, mu: float = 0.0, beta: float = 1.0) -> Expression:
    dim, batch_size = x.dim()
    return gumbel(*dim, mu=mu, beta=beta, batch_size=batch_size)


def where(cond: Expression, x: Expression, y: Expression) -> Expression:
    return dy.cmult(cond, x) + dy.cmult(1.0 - cond, y)


if __name__ == '__main__':
    a = dy.inputTensor([[1, 2, 3], [2, 3, 4], ])
    print(f'a :: {a.dim()} => {a.value()}')

    b = diagonal(a)
    print(f'b :: {b.dim()} => {b.value()}')
