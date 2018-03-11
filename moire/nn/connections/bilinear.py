import dynet as dy

import moire
from moire import ParameterCollection, Expression
from moire import nn
from moire.nn.inits import Zero, GlorotUniform


class BiLinear(nn.Module):
    def __init__(self, pc: ParameterCollection,
                 in1_features: int, in2_features: int, out_features: int, bias: bool = True,
                 weight_initializer=GlorotUniform(), bias_initializer=Zero()) -> None:
        super(BiLinear, self).__init__(pc)

        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.bias = bias

        self.weight = self.add_param((in1_features * out_features, in2_features), weight_initializer)
        if bias:
            self.bias = self.add_param((out_features,), bias_initializer)

    def __repr__(self):
        return f'{self.__class__.__name__} ({self.in1_features}, {self.in2_features} -> {self.out_features})'

    def __call__(self, x1: Expression, x2: Expression) -> Expression:
        weight = self.weight.expr(moire.config.train)
        u = dy.transpose(dy.reshape(weight * x2, (self.in1_features, self.out_features)))
        if self.bias:
            bias = self.bias.expr(moire.config.train)
            return dy.affine_transform([bias, u, x1])
        return u * x1
