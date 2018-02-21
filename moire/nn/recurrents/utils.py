from typing import List

from moire import Expression


def ensure_tuple(*args):
    return args


def scan(rnn, xs: List[Expression], *hiddens):
    for x in xs:
        hiddens = ensure_tuple(rnn(x, *hiddens))
        yield hiddens