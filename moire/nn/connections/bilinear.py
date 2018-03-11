import dynet as dy

import nn
from moire import ParameterCollection, Expression


class BiLinear(nn.Module):
    def __init__(self, pc: ParameterCollection,
                 in1_features: int, in2_features: int, out_features: int, bias: bool = True) -> None:
        super(BiLinear, self).__init__(pc)

        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.bias = bias

        self.weight = self.pc.add_parameters((in1_features * out_features, in2_features))
        if bias:
            self.bias = self.pc.add_parameters((out_features,))

    def __repr__(self):
        return f'{self.__class__.__name__} ({self.in1_features}, {self.in2_features} -> {self.out_features})'

    def __call__(self, x1: Expression, x2: Expression) -> Expression:
        weight = self.weight.expr(self.training)
        u = dy.transpose(dy.reshape(weight * x2, (self.in1_features, self.out_features)))
        if self.bias:
            bias = self.bias.expr(self.training)
            return dy.affine_transform([bias, u, x1])
        return u * x1