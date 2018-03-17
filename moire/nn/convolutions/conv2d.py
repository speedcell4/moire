from typing import Tuple, Union, Optional

import dynet as dy
from dynet import Model

import moire
from moire import nn, Expression, ParameterCollection, normal
from moire.nn.initializers import Xavier, Zero


class Conv2d(nn.Module):
    def __init__(self, pc: ParameterCollection, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride_size: Union[int, Tuple[int, int]], pad_zero: bool = False,
                 kernel_initializer=Xavier(), bias_initializer=Zero(), use_bias: bool = True) -> None:
        super(Conv2d, self).__init__(pc)

        self.is_valid = not pad_zero

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = list(kernel_size)
        if isinstance(stride_size, int):
            stride_size = (stride_size, stride_size)
        self.stride_size = list(stride_size)
        self.use_bias = use_bias

        self.f = self.add_param((*self.kernel_size, in_channels, out_channels), kernel_initializer)
        if use_bias:
            self.b = self.add_param((out_channels,), bias_initializer)

    def __repr__(self):
        k1, k2 = self.kernel_size
        s1, s2 = self.stride_size
        return f'{self.__class__.__name__} ({k1} x {k2} @ {s1} x {s2})'

    def __call__(self, x: Expression) -> Expression:
        """
        :param x: [Height x Width x nChannels, nBatches]
        :return: [Height x Width x nChannels, nBatches]
        """

        f = self.f.expr(moire.config.train)
        if self.use_bias:
            b = self.b.expr(moire.config.train)
            return dy.conv2d_bias(x, f, b, self.stride_size, is_valid=self.is_valid)
        return dy.conv2d(x, f, self.stride_size, is_valid=self.is_valid)


class MaxPooling2d(nn.Function):
    def __init__(self, kernel_size: Union[Optional[int], Tuple[Optional[int], Optional[int]]],
                 stride_size: Union[int, Tuple[int, int]], pad_zero: bool = False) -> None:
        super(MaxPooling2d, self).__init__()

        self.is_valid = not pad_zero

        if isinstance(kernel_size, int) or kernel_size is None:
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = list(kernel_size)
        if isinstance(stride_size, int):
            stride_size = (stride_size, stride_size)
        self.stride_size = list(stride_size)

    def __repr__(self) -> str:
        k1, k2 = self.kernel_size
        s1, s2 = self.stride_size
        return f'{self.__class__.__name__} ({k1} x {k2} @ {s1} x {s2})'

    def __call__(self, x: Expression) -> Expression:
        """
        :param x: [Height x Width x nChannels, nBatches]
        :return: [Height x Width x nChannels, nBatches]
        """
        (height0, width0) = self.kernel_size
        (height1, width1, num_channels), num_batches = x.dim()
        kernel_size = (height0 or height1, width0 or width1)
        return dy.maxpooling2d(x, kernel_size, self.stride_size, self.is_valid)


if __name__ == '__main__':
    in_channels = 2
    out_channels = 10
    conv2d = Conv2d(Model(), in_channels, out_channels, 3, 1, use_bias=False)
    maxpool2d = MaxPooling2d((3, None), 1, pad_zero=True)
    dy.renew_cg(True, True)

    x = normal(7, 7, in_channels)

    print(conv2d)
    print(maxpool2d)

    print(x.dim())
    y = conv2d(x)
    print(y.dim())

    print(maxpool2d(y).dim())
