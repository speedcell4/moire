from typing import List

import dynet as dy
import numpy as np

import moire
from moire.nn import initializers
from moire import Expression, ParameterCollection, nn, where
from moire.array import full_like

__all__ = [
    'sigmoid', 'softsign', 'hard_sigmoid',
    'softmax', 'log_softmax',
    'LogSoftmax',
    'MaskedLogSoftmax', 'RestrictedLogSoftmax',
]

sigmoid = dy.logistic
softsign = dy.softsign


def clip(x: Expression, a_min: float, a_max: float) -> Expression:
    a_min = full_like(x, a_min)
    a_max = full_like(x, a_max)
    return dy.bmax(a_min, dy.bmin(a_max, x))


def hard_sigmoid(x: Expression, slope: float = 0.2, offset: float = 0.5,
                 a_min: float = 0.0, a_max: float = 1.0) -> Expression:
    return clip(x * slope + offset, a_min, a_max)


def softmax(x: Expression, axis: int = 1) -> Expression:
    return dy.softmax(x, d=axis)


def softmax_mul(A: Expression, B: Expression, bias: Expression = None, scale: bool = True) -> Expression:
    (A_d0, A_d1), _ = A.dim()
    (B_d0, B_d1), _ = B.dim()
    assert A_d1 == B_d0, f'{A.dim()} is not capable with {B.dim()} for multiplication'

    if bias is not None:
        Z = dy.affine_transform([bias, A, B])
    else:
        Z = A * B

    if scale:
        return softmax(Z / A_d1, axis=1)
    else:
        return softmax(Z, axis=1)


class LogSoftmax(nn.Module):
    def __init__(self, pc: ParameterCollection, nb_classes: int, restrict=None) -> None:
        super(LogSoftmax, self).__init__(pc)

        self.nb_classes = nb_classes
        self.restrict = restrict

        if restrict is not None:
            array = np.full((nb_classes,), fill_value=-float('inf'), dtype=np.float32)
            for ix in restrict:
                array[ix] = np.float32(1.0)
            self.mask = self.add_param((nb_classes,), initializers.NumpyInitializer(array, np.float32))

    def __repr__(self) -> str:
        if self.restrict is not None:
            return f'<{self.__class__.__name__} @ {self.restrict}>'
        else:
            return f'<{self.__class__.__name__}>'

    def __call__(self, x: Expression) -> Expression:
        if self.restrict is not None:
            x = dy.cmult(x, self.mask.expr(moire.config.train))
        return dy.log_softmax(x)


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


if __name__ == '__main__':
    A = dy.inputTensor([[1, 2], [2, 3]])
    B = dy.inputTensor([[1, 2], [2, 3]])

    C = softmax_mul(A, B, None, True)
    D = A * B

    print(f'C :: {C.dim()} => {C.value()}')
    print(f'D :: {D.dim()} => {D.value()}')
