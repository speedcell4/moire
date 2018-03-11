from typing import List

import dynet as dy
import numpy as np

import moire
from moire.array import full_like
from moire import ParameterCollection, nn, Expression, where

__all__ = [
    'sigmoid', 'hard_sigmoid',
    'log_softmax',
    'MaskedLogSoftmax', 'RestrictedLogSoftmax',
]

sigmoid = dy.logistic


def clip(x: Expression, a_min: float, a_max: float) -> Expression:
    a_min = full_like(x, a_min)
    a_max = full_like(x, a_max)
    return dy.bmax(a_min, dy.bmin(a_max, x))


def hard_sigmoid(x: Expression) -> Expression:
    return clip(x * 0.2 + 0.5, 0.0, 1.0)


def log_softmax(x: Expression, restrict=None) -> Expression:
    if restrict is None or moire.config.device == 'CPU':
        return dy.log_softmax(x, restrict)
    y = dy.log_softmax(dy.to_device(x, 'CPU'), restrict=restrict)
    return dy.to_device(y, moire.config.device)


class MaskedLogSoftmax(nn.Module):
    def __init__(self, pc: ParameterCollection, mask: List[int], full_value: float = 0.0):
        super(MaskedLogSoftmax, self).__init__(pc)

        dim = (len(mask),)
        self.mask = self.pc.add_parameters(dim, init=dy.NumpyInitializer(np.array(mask)))
        self.full = self.pc.add_parameters(dim, init=dy.ConstInitializer(full_value))
        self.zero = self.pc.add_parameters(dim, init=dy.ConstInitializer(0.0))

    def __repr__(self):
        return f'<{self.__class__.__name__} :: {self.mask}>'

    def __call__(self, x: Expression) -> Expression:
        mask = self.mask.expr(False)
        full = self.mask.expr(False)
        zero = self.mask.expr(False)

        y = dy.cmult(mask, x)
        exp_x = where(mask, dy.exp(y - dy.max_dim(y)), zero)
        return where(mask, x - dy.log(dy.sum_elems(exp_x)), full)


class RestrictedLogSoftmax(MaskedLogSoftmax):
    def __init__(self, pc: ParameterCollection, n: int, restricts: List[int], full_value: float = 0.0):
        mask = [0] * n
        for i in restricts:
            mask[i] = 1
        self.restricts = restricts
        super(RestrictedLogSoftmax, self).__init__(pc, mask, full_value)

    def __repr__(self):
        return f'<{self.__class__.__name__} :: {self.restricts}>'
