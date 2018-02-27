import dynet as dy

from moire import nn, Expression, ParameterCollection, Activation
from moire.nn.utils import compute_hidden_size


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


class MLP(nn.Module):
    def __init__(self, pc: ParameterCollection, num_layers: int,
                 in_features: int, out_features: int, hidden_features: int = None,
                 bias: bool = True, nonlinear: Activation = dy.tanh) -> None:
        super(MLP, self).__init__(pc)

        if hidden_features is None:
            hidden_features = compute_hidden_size(in_features, out_features)

        self.num_layers = num_layers
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.bias = bias
        self.nonlinear = nonlinear

        if num_layers == 1:
            self.layer0 = Linear(self.pc, in_features, out_features)
        else:
            self.layer0 = Linear(self.pc, in_features, hidden_features)
            for ix in range(1, num_layers - 1):
                setattr(self, f'layer{ix}', Linear(self.pc, hidden_features, hidden_features))
            setattr(self, f'layer{num_layers - 1}', Linear(self.pc, hidden_features, out_features))

    def __repr__(self):
        return f'{self.__class__.__name__} ({self.num_layers} @ {self.in_features} -> ' \
               f'{self.hidden_features} -> {self.out_features})'

    # TODO nonlinear activations
    def __call__(self, x: Expression) -> Expression:
        for ix in range(self.num_layers - 1):
            x = self.nonlinear(getattr(self, f'layer{ix}')(x))
        return getattr(self, f'layer{self.num_layers - 1}')(x)


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


if __name__ == '__main__':
    fc = MLP(ParameterCollection(), 3, 4, 5)
    dy.renew_cg(True, True)

    # x1 = dy.inputVector([1, 2, 3])
    x2 = dy.inputVector([1, 2, 3, 4])
    y = fc(x2)

    print(y.dim())
    print(y.value())

    print(fc)
