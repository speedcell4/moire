import dynet as dy

import moire
from moire import nn, ParameterCollection, Expression
from moire.nn.inits import GlorotUniform, Zero


class Linear(nn.Module):
    def __init__(self, pc: ParameterCollection, in_features: int, out_features: int,
                 weight_initializer=GlorotUniform(), bias_initializer=Zero(), use_bias: bool = True) -> None:
        super(Linear, self).__init__(pc)

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

        self.W = self.add_param((out_features, in_features), weight_initializer)
        if use_bias:
            self.b = self.add_param((out_features,), bias_initializer)

    def __repr__(self):
        return f'{self.__class__.__name__} ({self.in_features} -> {self.out_features})'

    def __call__(self, x: Expression) -> Expression:
        W = self.W.expr(moire.config.train)
        if self.use_bias:
            b = self.b.expr(moire.config.train)
            return dy.affine_transform([b, W, x])
        return W * x


if __name__ == '__main__':
    pc = ParameterCollection()

    fc = Linear(pc, 4, 3)
    dy.renew_cg()

    x = dy.inputVector([1, 2, 3, 4])
    y = fc(x)

    print(y.value())
