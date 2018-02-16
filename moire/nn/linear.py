import dynet as dy

from moire import nn, Expression, ParameterCollection

__all__ = [
    'Linear',
]


# TODO initializer
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


if __name__ == '__main__':
    fc = Linear(ParameterCollection(), 3, 5, False)
    dy.renew_cg(True, True)

    x = dy.inputVector([1, 2, 3])
    y = fc(x)

    print(y.dim())
    print(y.value())

    print(fc)
