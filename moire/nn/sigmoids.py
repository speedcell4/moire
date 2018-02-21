from typing import List

import dynet as dy
import numpy as np

from moire import ParameterCollection, nn, Expression, where

__all__ = [
    'MaskedLogSoftmax',
]


class MaskedLogSoftmax(nn.Module):
    def __init__(self, pc: ParameterCollection, mask: List[int], full_value: float = 0.0):
        super(MaskedLogSoftmax, self).__init__(pc)

        dim = (len(mask),)
        self.mask = self.pc.add_parameters(dim, init=dy.NumpyInitializer(np.array(mask)))
        self.full = self.pc.add_parameters(dim, init=dy.ConstInitializer(full_value))
        self.zero = self.pc.add_parameters(dim, init=dy.ConstInitializer(0.0))

    @classmethod
    def from_restrict(cls, pc: ParameterCollection, n: int, restricts: List[int], full_value: float = 0.0):
        mask = [0] * n
        for i in restricts:
            mask[i] = 1
        return MaskedLogSoftmax(pc, mask, full_value)

    def __repr__(self):
        return f'<{self.__class__.__name__} :: {self.mask}>'

    def __call__(self, x: Expression) -> Expression:
        mask = self.mask.expr(False)
        full = self.mask.expr(False)
        zero = self.mask.expr(False)

        y = dy.cmult(mask, x)
        exp_x = where(mask, dy.exp(y - dy.max_dim(y)), zero)
        return where(mask, x - dy.log(dy.sum_elems(exp_x)), full)
