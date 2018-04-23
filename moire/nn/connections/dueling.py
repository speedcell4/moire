import functools

import dynet as dy

import moire
from moire import Expression, ParameterCollection
from moire import nn


# TODO check the last layer activation
class DuelingNetwork(nn.Module):
    def __init__(self, pc: ParameterCollection, num_layers: int, input_feature,
                 num_actions: int, reduce_strategy: str = 'avg') -> None:
        super().__init__(pc)
        assert reduce_strategy in ['avg', 'max']

        self.num_layers = num_layers
        self.input_feature = input_feature
        self.num_actions = num_actions

        self.v_function = nn.MLP(self.pc, num_layers, input_feature, 1)
        self.a_function = nn.MLP(self.pc, num_layers, input_feature, num_actions)

        if reduce_strategy == 'avg':
            self.reduce = functools.partial(dy.mean_dim, d=[0], b=False)
        else:
            self.reduce = dy.max_dim

    def __call__(self, x: Expression) -> Expression:
        v = self.v_function(x)
        a = self.a_function(x)
        return v + (a - self.reduce(a))


if __name__ == '__main__':
    fc = DuelingNetwork(ParameterCollection(), 2, 4, 3, 'max')
    dy.renew_cg()

    x = dy.inputVector([1, 2, 3, 4])
    moire.debug(fc(x).value())
