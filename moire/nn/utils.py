import numpy as np

__all__ = [
    'compute_hidden_size',
]


# check here
def compute_hidden_size(in_size: int, out_size: int) -> int:
    return int(np.maximum(in_size, out_size) + np.ceil(np.sqrt(in_size + out_size)))


if __name__ == '__main__':
    print(compute_hidden_size(4, 3))
