import dynet as dy

import moire
from moire import ParameterCollection, Expression
from moire import nn
from moire.nn.inits import Zero, GlorotUniform


class BiLinear(nn.Module):
    def __init__(self, pc: ParameterCollection,
                 in1_feature: int, in2_feature: int, out_feature: int, bias: bool = True,
                 weight_initializer=GlorotUniform(), bias_initializer=Zero()) -> None:
        super(BiLinear, self).__init__(pc)

        self.in1_feature = in1_feature
        self.in2_feature = in2_feature
        self.out_feature = out_feature
        self.bias = bias

        self.weight = self.add_param((in1_feature * out_feature, in2_feature), weight_initializer)
        if bias:
            self.bias = self.add_param((out_feature,), bias_initializer)

    def __repr__(self):
        return f'{self.__class__.__name__} ({self.in1_feature}, {self.in2_feature} -> {self.out_feature})'

    def __call__(self, x1: Expression, x2: Expression) -> Expression:
        weight = self.weight.expr(moire.config.train)
        u = dy.transpose(dy.reshape(weight * x2, (self.in1_feature, self.out_feature)))
        if self.bias:
            bias = self.bias.expr(moire.config.train)
            return dy.affine_transform([bias, u, x1])
        return u * x1
