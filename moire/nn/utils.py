from typing import Union

import numpy as np
from numpy.core.multiarray import ndarray

from moire import Expression

__all__ = [
    'compute_hidden_size',
]


# check here
def compute_hidden_size(in_size: int, out_size: int) -> int:
    return int(np.maximum(in_size, out_size) + np.ceil(np.sqrt(in_size + out_size)))


if __name__ == '__main__':
    print(compute_hidden_size(4, 3))


def to_numpy(value: Union[Expression, bool, int, float, ndarray]) -> ndarray:
    if isinstance(value, Expression):
        return value.npvalue()
    if isinstance(value, (bool, int, float)):
        return np.array([value], dtype=np.float32)
    return np.array(value, dtype=np.float32)