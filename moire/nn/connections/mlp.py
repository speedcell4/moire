from moire import ParameterCollection, Activation, Expression
from moire import nn
from moire.nn.connections.linear import Linear
from moire.nn.inits import GlorotUniform, calculate_gain
from moire.nn.thresholds import LeakyRelu
from moire.nn.utils import compute_hidden_size


class MLP(nn.Module):
    def __init__(self, pc: ParameterCollection, num_layers: int,
                 in_features: int, out_features: int, hidden_features: int = None,
                 use_bias: bool = True, nonlinear: Activation = LeakyRelu()) -> None:
        super(MLP, self).__init__(pc)

        if hidden_features is None:
            hidden_features = compute_hidden_size(in_features, out_features, )

        self.num_layers = num_layers
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.bias = use_bias
        self.nonlinear = nonlinear

        if num_layers == 1:
            self.layer0 = Linear(self.pc, in_features, out_features)
        else:
            weight_initializer = GlorotUniform(gain=calculate_gain(nonlinear))
            self.layer0 = Linear(self.pc, in_features=in_features, out_features=hidden_features,
                                 weight_initializer=weight_initializer)
            for ix in range(1, num_layers - 1):
                setattr(self, f'layer{ix}', Linear(self.pc, in_features=hidden_features, out_features=hidden_features,
                                                   weight_initializer=weight_initializer))
            setattr(self, f'layer{num_layers - 1}', Linear(self.pc, hidden_features, out_features))

    def __repr__(self):
        return f'{self.__class__.__name__} ({self.num_layers} @ {self.in_features} -> ' \
               f'{self.hidden_features} -> {self.out_features})'

    # TODO nonlinear activations
    def __call__(self, x: Expression) -> Expression:
        for ix in range(self.num_layers - 1):
            x = self.nonlinear(getattr(self, f'layer{ix}')(x))
        return getattr(self, f'layer{self.num_layers - 1}')(x)


if __name__ == '__main__':
    pass
