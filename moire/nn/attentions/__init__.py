import functools

import dynet as dy

import moire
from moire import nn
from moire import Expression


def check_attention_shape(func):
    @functools.wraps(func)
    def wrapper(Q: Expression, K: Expression, V: Expression, *args, **kwargs):
        (q_batch, q_size), _ = Q.dim()
        (k_batch, k_size), _ = K.dim()
        (v_batch, v_size), _ = V.dim()
        assert q_size == k_size, f'{q_size} != {k_size}'
        assert k_batch == v_batch, f'{k_batch} != {v_batch}'

        return func(Q, K, V, *args, **kwargs)

    return wrapper


@check_attention_shape
def dot_product_attention(Q: Expression, K: Expression, V: Expression, scale: bool = True) -> Expression:
    A = Q * dy.transpose(K)
    if scale:
        (_, d), _ = K.dim()
        return nn.softmax(A / d, axis=1) * V
    return nn.softmax(A, axis=1) * V


if __name__ == '__main__':
    dy.renew_cg()

    Q = moire.normal(4, 5)
    K = moire.normal(3, 5)
    V = moire.normal(3, 7)

    Z = dot_product_attention(Q, K, V)
    moire.debug(f'Z :: {Z.dim()} => {Z.value()}')
