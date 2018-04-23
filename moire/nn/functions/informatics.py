import dynet as dy

import moire
from moire import Expression


def entropy(p: Expression) -> Expression:
    return -dy.dot_product(p, dy.log(p))


if __name__ == '__main__':
    p = dy.softmax(dy.inputVector([1, 2, 3, 4]))
    moire.debug(entropy(p).value())
