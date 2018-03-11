import dynet as dy

import nn
from moire import ParameterCollection, Expression


class Linear(nn.Module):
    def __init__(self, pc: ParameterCollection, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__(pc)

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.W = self.pc.add_parameters((out_features, in_features))
        if bias:
            self.b = self.pc.add_parameters((out_features,))

    def __repr__(self):
        return f'{self.__class__.__name__} ({self.in_features} -> {self.out_features})'

    def __call__(self, x: Expression) -> Expression:
        W = self.W.expr(self.training)
        if self.bias:
            b = self.b.expr(self.training)
            return dy.affine_transform([b, W, x])
        return W * x