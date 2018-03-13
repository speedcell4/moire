from typing import List

import dynet as dy

from moire import Expression, ParameterCollection
from moire import nn
from moire.nn.inits import GlorotNormal, Orthogonal, Uniform, Zero, One
from moire.nn.sigmoids import sigmoid
from moire.nn.trigonometry import tanh


class LSTM(nn.Module):
    def __init__(self, pc: ParameterCollection, num_layers: int,
                 input_size: int, hidden_size: int,
                 dropout_ratio: float = None, zoneout_ratio: float = None,
                 activation=tanh, recurrent_activation=sigmoid,
                 kernel_initializer=GlorotNormal(), recurrent_initializer=Orthogonal(),
                 hidden_initializer=Uniform(), bias_initializer=Zero(), forget_bias_initializer=One()):
        super().__init__(pc)

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_ratio = dropout_ratio
        self.zoneout_ratio = zoneout_ratio
        self.rnn = dy.LSTMBuilder(num_layers, input_size, hidden_size, self.pc)

    def init_state(self):
        return self.rnn.initial_state()

    def transduce(self, xs: List[Expression]) -> List[Expression]:
        return self.rnn.initial_state().transduce(xs)

    def compress(self, xs: List[Expression]) -> Expression:
        if len(xs) == 0:
            return self.rnn.initial_state().output()
        return self.rnn.initial_state().transduce(xs)[-1]


class BiLSTM(nn.Module):
    def __init__(self, pc: ParameterCollection, num_layers: int,
                 input_size: int, output_size: int, merge_strategy: str,
                 dropout_ratio: float = None, zoneout_ratio: float = None,
                 activation=tanh, recurrent_activation=sigmoid,
                 kernel_initializer=GlorotNormal(), recurrent_initializer=Orthogonal(),
                 hidden_initializer=Uniform(), bias_initializer=Zero(), forget_bias_initializer=One()) -> None:
        super(BiLSTM, self).__init__(pc)
        assert merge_strategy in ['cat', 'avg', 'sum', 'max']

        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size
        self.dropout_ratio = dropout_ratio
        self.zoneout_ratio = zoneout_ratio

        self.merge_strategy = merge_strategy

        if merge_strategy == 'cat':
            forward_size = output_size // 2
            backward_size = output_size - forward_size
            self.merge = dy.concatenate
        elif merge_strategy == 'avg':
            forward_size = backward_size = output_size
            self.merge = dy.average
        elif merge_strategy == 'sum':
            forward_size = backward_size = output_size
            self.merge = dy.esum
        elif merge_strategy == 'max':
            forward_size = backward_size = output_size
            self.merge = dy.emax
        else:
            raise NotImplementedError(f'no such merge_strategy :: {merge_strategy}')

        self.f = LSTM(self.pc, num_layers, input_size, forward_size,
                      dropout_ratio, zoneout_ratio, activation, recurrent_activation,
                      kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer,
                      hidden_initializer=hidden_initializer,
                      bias_initializer=bias_initializer, forget_bias_initializer=forget_bias_initializer)
        self.b = LSTM(self.pc, num_layers, input_size, backward_size,
                      dropout_ratio, zoneout_ratio, activation, recurrent_activation,
                      kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer,
                      hidden_initializer=hidden_initializer,
                      bias_initializer=bias_initializer, forget_bias_initializer=forget_bias_initializer)

    def transduce(self, xs: List[Expression],
                  fhtm1s: List[Expression] = None, fctm1s: List[Expression] = None,
                  bhtm1s: List[Expression] = None, bctm1s: List[Expression] = None) -> List[Expression]:
        fs = self.f.transduce(xs)
        bs = self.b.transduce(xs[::-1])[::-1]
        return [self.merge([f, b]) for f, b in zip(fs, bs)]

    def compress(self, xs: List[Expression],
                 fhtm1s: List[Expression] = None, fctm1s: List[Expression] = None,
                 bhtm1s: List[Expression] = None, bctm1s: List[Expression] = None) -> Expression:
        if len(xs) == 0:
            f = self.f.init_state().hts[-1]
            b = self.b.init_state().hts[-1]
        else:
            f = self.f.transduce(xs)[-1]
            b = self.b.transduce(xs[::-1])[0]
        return self.merge([f, b])


if __name__ == '__main__':
    rnn = BiLSTM(ParameterCollection(), 2, 3, 4, 'avg')

    xs = [
        dy.inputVector([1, 2, 3]),
        dy.inputVector([1, 2, 3]),
        dy.inputVector([1, 2, 3]),
    ]

    y = rnn.compress(xs)
    print(f'y :: {y.dim()} => {y.value()}')

    for z in rnn.transduce(xs):
        print(f'z :: {z.dim()} => {z.value()}')
