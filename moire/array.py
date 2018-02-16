import dynet as dy

from moire import Expression

__all__ = [
    'zeros', 'ones', 'full', 'normal', 'bernoulli', 'uniform', 'gumbel',
    'zeros_like', 'ones_like', 'full_like', 'normal_like', 'bernoulli_like', 'uniform_like', 'gumbel_like',
    'where',
]


def zeros(*dim, batch_size: int = 1) -> Expression:
    return dy.zeros(dim, batch_size=batch_size)


def zeros_like(x: Expression) -> Expression:
    dim, batch_size = x.dim()
    return dy.zeros(dim, batch_size=batch_size)


def ones(*dim, batch_size: int = 1) -> Expression:
    return dy.ones(dim, batch_size=batch_size)


def ones_like(x: Expression) -> Expression:
    dim, batch_size = x.dim()
    return dy.ones(dim, batch_size=batch_size)


def full(*dim, value, batch_size: int = 1) -> Expression:
    return dy.constant(dim, value, batch_size=batch_size)


def full_like(x: Expression, value) -> Expression:
    dim, batch_size = x.dim()
    return dy.constant(dim, value, batch_size=batch_size)


def normal(*dim, mean: float = 0.0, stddev: float = 1.0, batch_size: int = 1) -> Expression:
    return dy.random_normal(dim, mean=mean, stddev=stddev, batch_size=batch_size)


def normal_like(x: Expression, mean: float = 0.0, stddev: float = 1.0) -> Expression:
    dim, batch_size = x.dim()
    return dy.random_normal(dim, mean=mean, stddev=stddev, batch_size=batch_size)


def bernoulli(*dim, p: float, scale: float = 1.0, batch_size: int = 1) -> Expression:
    return dy.random_bernoulli(dim, p=p, scale=scale, batch_size=batch_size)


def bernoulli_like(x: Expression, p: float, scale: float = 1.0) -> Expression:
    dim, batch_size = x.dim()
    return dy.random_bernoulli(dim, p=p, scale=scale, batch_size=batch_size)


def uniform(*dim, low: float, high: float, batch_size: int = 1) -> Expression:
    return dy.random_uniform(dim, left=low, right=high, batch_size=batch_size)


def uniform_like(x: Expression, low: float, high: float) -> Expression:
    dim, batch_size = x.dim()
    return dy.random_uniform(dim, left=low, right=high, batch_size=batch_size)


def gumbel(*dim, mu: float = 0.0, beta: float = 1.0, batch_size: int = 1) -> Expression:
    return dy.random_gumbel(dim, mu=mu, beta=beta, batch_size=batch_size)


def gumbel_like(x: Expression, mu: float = 0.0, beta: float = 1.0) -> Expression:
    dim, batch_size = x.dim()
    return dy.random_gumbel(dim, mu=mu, beta=beta, batch_size=batch_size)


def where(cond: Expression, x: Expression, y: Expression) -> Expression:
    return dy.cmult(cond, x) + dy.cmult(1.0 - cond, y)


if __name__ == '__main__':
    cond = dy.inputVector([1, 0, 0, 1])
    x = dy.inputVector([1, 2, 3, 4])
    y = dy.inputVector([6, 7, 8, 9])
    print(where(cond, x, y).value())
