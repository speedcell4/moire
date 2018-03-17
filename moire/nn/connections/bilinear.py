import warnings

import dynet as dy

import moire
from moire import Expression, ParameterCollection
from moire import nn
from moire.nn.inits import GlorotUniform, Zero


class BiLinear(nn.Module):
    def __init__(self, pc: ParameterCollection,
                 in1_feature: int, in2_feature: int, out_feature: int,
                 use_u: bool = True, use_v: bool = True, use_bias: bool = True,
                 weight_initializer=GlorotUniform(), bias_initializer=Zero()) -> None:
        super(BiLinear, self).__init__(pc)
        warnings.warn(f'{self.__class__.__name__} has not been tested on GPU', FutureWarning)

        self.in1_feature = in1_feature
        self.in2_feature = in2_feature
        self.out_feature = out_feature

        self.use_u = use_u
        self.use_v = use_v
        self.use_bias = use_bias

        self.W = self.add_param((out_feature, in2_feature, in1_feature), weight_initializer)
        if use_u:
            self.U = self.add_param((out_feature, in1_feature), weight_initializer)
        if use_v:
            self.V = self.add_param((out_feature, in2_feature), weight_initializer)
        if use_bias:
            self.bias = self.add_param((out_feature,), bias_initializer)

    def __repr__(self):
        return f'{self.__class__.__name__} ({self.in1_feature}, {self.in2_feature} -> {self.out_feature})'

    def __call__(self, x1: Expression, x2: Expression) -> Expression:
        W = self.W.expr(moire.config.train)
        if self.use_bias:
            b = self.bias.expr(moire.config.train)
            xs = [dy.contract3d_1d_1d_bias(W, x1, x2, b)]
        else:
            xs = [dy.contract3d_1d_1d(W, x1, x2)]
        if self.use_u:
            u = self.U.expr(moire.config.train)
            xs.extend([u, x1])
        if self.use_v:
            v = self.V.expr(moire.config.train)
            xs.extend([v, x2])
        return dy.affine_transform(xs)
