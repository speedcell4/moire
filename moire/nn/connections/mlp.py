from itertools import chain, repeat

import dynet as dy
from moire import Activation, Expression, ParameterCollection, nn
from moire.nn.connections.linear import Linear
from moire.nn.initializers import GlorotUniform, Zero, calculate_gain
from moire.nn.functions import LeakyRelu, linear
from moire.nn.utils import compute_hidden_size

__all__ = [
    'MLP',
]


class MLP(nn.Module):
    def __init__(self, pc: ParameterCollection, num_layers: int,
                 in_feature: int, out_feature: int, hidden_feature: int = None, use_bias: bool = True,
                 hidden_activation: Activation = LeakyRelu(), output_activation: Activation = linear,
                 weight_initializer=None, output_weight_initializer=None, bias_initializer=Zero()) -> None:
        super(MLP, self).__init__(pc)

        if hidden_feature is None:
            hidden_feature = compute_hidden_size(in_feature, out_feature)

        self.num_layers = num_layers
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.hidden_feature = hidden_feature
        self.bias = use_bias
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        if weight_initializer is None:
            weight_initializer = GlorotUniform(gain=calculate_gain(hidden_activation))
        if output_weight_initializer is None:
            output_weight_initializer = GlorotUniform(gain=calculate_gain(output_activation))

        in_features = chain([in_feature], repeat(hidden_feature, num_layers - 1))
        hidden_features = chain(repeat(hidden_feature, num_layers - 1), [out_feature])
        weight_initializers = chain(repeat(weight_initializer, num_layers - 1), [output_weight_initializer])
        activations = chain(repeat(hidden_activation, num_layers - 1), [output_activation])

        parameters = zip(in_features, hidden_features, weight_initializers, activations)
        for ix, (in_feature, hidden_feature, weight_initializer, activation) in enumerate(parameters):
            layer = Linear(self.pc, in_feature, hidden_feature, weight_initializer, bias_initializer, use_bias)
            setattr(self, f'layer{ix}', layer)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__} ({self.num_layers} layers :: {self.in_feature} -> ' \
               f'{self.hidden_feature} -> {self.out_feature})'

    def __call__(self, x: Expression) -> Expression:
        for ix in range(self.num_layers - 1):
            x = self.hidden_activation(getattr(self, f'layer{ix}')(x))
        return self.output_activation(getattr(self, f'layer{self.num_layers - 1}')(x))


if __name__ == '__main__':
    fc = MLP(ParameterCollection(), num_layers=3, in_feature=4, out_feature=5)
    dy.renew_cg()

    x = dy.inputVector([1, 2, 3, 4])
    y = fc(x)

    print(fc)
    print(f'y :: {y.dim()} => {y.value()}')
