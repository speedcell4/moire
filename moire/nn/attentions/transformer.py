import dynet as dy
import numpy as np

import moire
from moire import nn
from moire.nn import initializers
from moire.nn.normalizations import LayerNorm
from moire import Expression, ParameterCollection


def dot_product_attention(Q: Expression, K: Expression, V: Expression) -> Expression:
    return dy.softmax(Q * dy.transpose(K) / np.sqrt(K.dim()[1]), 1) * V


class MultiHead(nn.Module):
    def __init__(self, pc: ParameterCollection, nb_heads: int, model_size: int,
                 key_size: int = None, value_size: int = None,
                 attention=dot_product_attention,
                 initializer=initializers.GlorotNormal()) -> None:
        super().__init__(pc)

        if key_size is None:
            key_size = model_size
        if value_size is None:
            value_size = model_size

        self.nb_heads = nb_heads
        self.query_size = key_size
        self.memory_size = value_size
        self.model_size = model_size

        for ix in range(self.nb_heads):
            setattr(self, f'Q{ix}', self.add_param((key_size, model_size // nb_heads), initializer))
            setattr(self, f'K{ix}', self.add_param((key_size, model_size // nb_heads), initializer))
            setattr(self, f'V{ix}', self.add_param((value_size, model_size // nb_heads), initializer))
        self.W = self.add_param((model_size, model_size), initializer)
        self.attention = attention

    def __call__(self, Q: Expression, K: Expression, V: Expression) -> Expression:
        c = []
        for ix in range(self.nb_heads):
            q = Q * getattr(self, f'Q{ix}').expr(moire.config.train)
            k = K * getattr(self, f'K{ix}').expr(moire.config.train)
            v = V * getattr(self, f'V{ix}').expr(moire.config.train)
            c.append(self.attention(q, k, v))
        return dy.concatenate(c, 1) * self.W.expr(moire.config.train)


class TransformerEncoder(nn.Module):
    def __init__(self, pc: ParameterCollection, nb_layers: int, nb_heads: int, model_size: int, key_size: int = None,
                 value_size: int = None, attention=dot_product_attention, initializer=initializers.GlorotNormal()):
        super(TransformerEncoder, self).__init__(pc)

        self.nb_layers = nb_layers

        self.ln0_0 = LayerNorm(self.pc, model_size)
        self.ln0_1 = LayerNorm(self.pc, model_size)
        self.feed0 = nn.Linear(self.pc, model_size, model_size)
        self.head0 = MultiHead(
            self.pc, nb_heads, model_size, key_size, value_size, attention, initializer)
        for ix in range(1, self.nb_layers):
            setattr(self, f'ln{ix}_0', LayerNorm(self.pc, model_size))
            setattr(self, f'ln{ix}_1', LayerNorm(self.pc, model_size))
            setattr(self, f'feed{ix}', nn.Linear(self.pc, model_size, model_size))
            setattr(self, f'head{ix}', MultiHead(
                self.pc, nb_heads, model_size, model_size, model_size, attention, initializer))

    def __call__(self, x: Expression) -> Expression:
        for ix in range(1, self.nb_layers):
            ln0 = getattr(self, f'ln{ix}_0')
            ln1 = getattr(self, f'ln{ix}_1')
            x = getattr(self, f'head{ix}')(x, x, x) + x
            x = getattr(self, f'feed{ix}')(x) + x
        return x


if __name__ == '__main__':
    head = TransformerEncoder(ParameterCollection(), 4, 2, 10)

    dy.renew_cg()

    q = moire.normal(4, 10)
    z = head(q)
    print(f'z :: {z.dim()} => {z.value()}')
