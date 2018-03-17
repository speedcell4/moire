import dynet as dy

import moire
from moire import nn, ParameterCollection, Expression
from moire.nn.initializers import GlorotUniform, Zero


class Linear(nn.Module):
    def __init__(self, pc: ParameterCollection, in_feature: int, out_feature: int,
                 weight_initializer=GlorotUniform(), bias_initializer=Zero(), use_bias: bool = True) -> None:
        super(Linear, self).__init__(pc)

        self.in_feature = in_feature
        self.out_feature = out_feature
        self.use_bias = use_bias

        self.W = self.add_param((out_feature, in_feature), weight_initializer)
        if not use_bias:
            bias_initializer = Zero()
        self.b = self.add_param((out_feature,), bias_initializer)

    def __repr__(self):
        return f'{self.__class__.__name__} ({self.in_feature} -> {self.out_feature})'

    def __call__(self, x: Expression) -> Expression:
        W = self.W.expr(moire.config.train)
        b = self.b.expr(self.use_bias and moire.config.train)
        return dy.affine_transform([b, W, x])


if __name__ == '__main__':
    fc = Linear(ParameterCollection(), 4, 3)
    dy.renew_cg()

    x = dy.inputVector([1, 2, 3, 4])
    y = fc(x)

    print(f'y :: {y.dim()} => {y.value()}')
