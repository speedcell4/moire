import dynet as dy

import moire
from moire import Expression, ParameterCollection
from moire import nn
from moire.nn.inits import GlorotUniform, Zero


class AffineTransform(nn.Module):
    def __init__(self, pc: ParameterCollection, *in_features: int, out_feature: int, separate_weights: bool = True,
                 weight_initializer=GlorotUniform(), bias_initializer=Zero(), use_bias: bool = True) -> None:
        super(AffineTransform, self).__init__(pc)

        self.in_features = in_features
        self.out_feature = out_feature
        self.separate_weights = separate_weights
        self.use_bias = use_bias

        if self.separate_weights:
            for ix, in_feature in enumerate(in_features):
                setattr(self, f'W{ix}', self.add_param((out_feature, in_feature), weight_initializer))
        else:
            self.W = self.add_param((out_feature, sum(in_features)), weight_initializer)

        if not use_bias:
            bias_initializer = Zero()
        self.b = self.add_param((out_feature,), bias_initializer)

    def __call__(self, *xs: Expression) -> Expression:
        zs = [self.b.expr(self.use_bias and moire.config.train)]

        if self.separate_weights:
            for ix, x in enumerate(xs):
                zs.extend((getattr(self, f'W{ix}').expr(moire.config.train), x))
        else:
            zs.extend((self.W.expr(moire.config.train), dy.concatenate(list(xs))))
        return dy.affine_transform(zs)


if __name__ == '__main__':
    pc = ParameterCollection()
    affine_transform = AffineTransform(
        pc, 3, 4, 5, separate_weights=False, out_feature=6, use_bias=True)

    dy.renew_cg()

    x1 = dy.inputVector([1, 2, 3])
    x2 = dy.inputVector([1, 2, 3, 4])
    x3 = dy.inputVector([1, 2, 3, 4, 5])

    y = affine_transform(x1, x2, x3)

    print(f'y :: {y.dim()} => {y.value()}')
