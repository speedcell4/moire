from moire import ParameterCollection, Activation, Expression
from moire import nn
from moire.nn.connections.linear import Linear
from moire.nn.inits import GlorotUniform, calculate_gain
from moire.nn.thresholds import LeakyRelu
from moire.nn.utils import compute_hidden_size


class MLP(nn.Module):
    def __init__(self, pc: ParameterCollection, num_layers: int,
                 in_feature: int, out_feature: int, hidden_feature: int = None,
                 use_bias: bool = True, nonlinear: Activation = LeakyRelu()) -> None:
        super(MLP, self).__init__(pc)

        if hidden_feature is None:
            hidden_feature = compute_hidden_size(in_feature, out_feature, )

        self.num_layers = num_layers
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.hidden_feature = hidden_feature
        self.bias = use_bias
        self.nonlinear = nonlinear

        if num_layers == 1:
            self.layer0 = Linear(self.pc, in_feature, out_feature)
        else:
            weight_initializer = GlorotUniform(gain=calculate_gain(nonlinear))
            self.layer0 = Linear(self.pc, in_feature=in_feature, out_feature=hidden_feature,
                                 weight_initializer=weight_initializer)
            for ix in range(1, num_layers - 1):
                setattr(self, f'layer{ix}', Linear(self.pc, in_feature=hidden_feature, out_feature=hidden_feature,
                                                   weight_initializer=weight_initializer))
            setattr(self, f'layer{num_layers - 1}', Linear(self.pc, hidden_feature, out_feature))

    def __repr__(self):
        return f'{self.__class__.__name__} ({self.num_layers} @ {self.in_feature} -> ' \
               f'{self.hidden_feature} -> {self.out_feature})'

    def __call__(self, x: Expression) -> Expression:
        for ix in range(self.num_layers - 1):
            x = self.nonlinear(getattr(self, f'layer{ix}')(x))
        return getattr(self, f'layer{self.num_layers - 1}')(x)
