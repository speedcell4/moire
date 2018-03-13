import dynet as dy

from moire import Expression


def entropy(p: Expression) -> Expression:
    return -dy.dot_product(p, dy.log(p))


if __name__ == '__main__':
    p = dy.softmax(dy.inputVector([1, 2, 3, 4]))
    print(entropy(p).value())
