import dynet as dy

import moire
from moire import nn
from moire.nn import initializers
from moire import Expression, ParameterCollection

__all__ = [
    'LayerNorm'
]


class LayerNorm(nn.Module):
    def __init__(self, pc: ParameterCollection, in_size: int,
                 initializer_gamma=initializers.One(), initializer_beta=initializers.Zero()) -> None:
        super().__init__(pc)

        self.in_size = in_size
        self.gamma = self.add_param((in_size,), initializer_gamma)
        self.beta = self.add_param((in_size,), initializer_beta)

    def __call__(self, x: Expression) -> Expression:
        gamma = self.gamma.expr(moire.config.train)
        beta = self.beta.expr(moire.config.train)
        return dy.layer_norm(x, gamma, beta)


if __name__ == '__main__':
    layer_norm = LayerNorm(ParameterCollection(), 6)
    dy.renew_cg()

    x = moire.normal(6, )
    z = layer_norm(x)
    print(f'x :: {x.dim()} => {x.value()}')
    print(f'z :: {z.dim()} => {z.value()}')
