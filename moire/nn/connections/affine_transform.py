import dynet as dy

import moire
from moire import Expression, ParameterCollection
from moire import nn
from moire.nn.inits import GlorotUniform, Zero


class AffineTransform(nn.Module):
    def __init__(self, pc: ParameterCollection, *in_features: int, out_feature: int,
                 weight_initializer=GlorotUniform(), bias_initializer=Zero(), use_bias: bool = True) -> None:
        super(AffineTransform, self).__init__(pc)

        self.in_features = in_features
        self.out_feature = out_feature
        self.use_bias = use_bias

        for ix, in_feature in enumerate(in_features):
            setattr(self, f'W{ix}', self.add_param((out_feature, in_feature), weight_initializer))

        if not use_bias:
            bias_initializer = Zero()
        self.b = self.add_param((out_feature,), bias_initializer)

    def __call__(self, *xs: Expression) -> Expression:
        zs = [self.b.expr(self.use_bias and moire.config.train)]
        for ix, x in enumerate(xs):
            zs.extend((getattr(self, f'W{ix}').expr(moire.config.train), x))
        return dy.affine_transform(zs)


if __name__ == '__main__':
    affine_transform = AffineTransform(ParameterCollection(), 3, 4, 5, out_feature=6, use_bias=True)
    dy.renew_cg()

    x1 = dy.inputVector([1, 2, 3])
    x2 = dy.inputVector([1, 2, 3, 4])
    x3 = dy.inputVector([1, 2, 3, 4, 5])

    y0 = affine_transform()
    y1 = affine_transform(x1)
    y2 = affine_transform(x1, x2)
    y3 = affine_transform(x1, x2, x3)

    print(f'y0 :: {y0.dim()} => {y0.value()}')
    print(f'y1 :: {y1.dim()} => {y1.value()}')
    print(f'y2 :: {y2.dim()} => {y2.value()}')
    print(f'y3 :: {y3.dim()} => {y3.value()}')
