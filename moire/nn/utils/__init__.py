from typing import Union

import numpy as np
from numpy.core.multiarray import ndarray

import moire
from moire import Expression

__all__ = [
    'compute_hidden_size',
    'to_numpy',
]


# check here
def compute_hidden_size(in_size: int, out_size: int, min_size: int = None, max_size: int = None) -> int:
    vec_size = int(np.maximum(in_size, out_size) + np.ceil(np.sqrt(in_size + out_size)))
    if min_size is not None:
        vec_size = max(min_size, vec_size)
    if max_size is not None:
        vec_size = min(max_size, vec_size)
    return vec_size


def to_numpy(value: Union[Expression, bool, int, float, ndarray]) -> ndarray:
    if isinstance(value, Expression):
        return value.npvalue()
    if isinstance(value, (bool, int, float)):
        return np.array([value], dtype=np.float32)
    return np.array(value, dtype=np.float32)


if __name__ == '__main__':
    moire.debug(compute_hidden_size(4, 3))
