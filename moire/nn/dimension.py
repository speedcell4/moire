import dynet as dy

from moire import Expression

__all__ = [
    'squeeze', 'unsqueeze'
]


def squeeze(x: Expression, axis: int) -> Expression:
    shape, _ = x.dim()
    return dy.reshape(x, shape[:axis] + shape[axis + 1:])


def unsqueeze(x: Expression, axis: int) -> Expression:
    shape, _ = x.dim()
    return dy.reshape(x, shape[:axis] + (1,) + shape[axis:])


if __name__ == '__main__':
    x = dy.reshape(dy.inputVector(list(range(2 * 3 * 1 * 4))), (2, 3, 1, 4))
    print(f'x :: {x.dim()} => {x.value()}')
    y = squeeze(x, 2)
    print(f'y :: {y.dim()} => {y.value()}')
    z = unsqueeze(y, 2)
    print(f'z :: {z.dim()} => {z.value()}')
