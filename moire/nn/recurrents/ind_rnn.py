from typing import List
import itertools

import dynet as dy

import moire
from moire import nn
from moire.nn.functions import LeakyRelu
from moire.nn.initializers import GlorotNormal, One, Uniform, Zero, calculate_gain
from moire import Expression, ParameterCollection


# TODO clip recurrent

class IndRNNCell(nn.Module):
    def __init__(self, pc: ParameterCollection,
                 input_size: int, hidden_size: int, activation=LeakyRelu(0.2),
                 W_initializer=None, u_initializer=One(), b_initializer=Zero(), h_initializer=Uniform()) -> None:
        super().__init__(pc)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation

        if W_initializer is None:
            W_initializer = GlorotNormal(calculate_gain(activation))

        self.W = self.add_param((hidden_size, input_size), W_initializer)
        self.u = self.add_param((hidden_size,), u_initializer)
        self.b = self.add_param((hidden_size,), b_initializer)
        self.h = self.add_param((hidden_size,), h_initializer)

    def __call__(self, x: Expression, htm1: Expression = None) -> Expression:
        W = self.W.expr(moire.config.train)
        u = self.u.expr(moire.config.train)
        b = self.b.expr(moire.config.train)

        if htm1 is None:
            htm1 = self.h.expr(moire.config.train)

        return self.activation(W * x + dy.cmult(u, htm1) + b)


class LSTMState(object):
    __slots__ = ('hts', 'step')

    def __init__(self, hts: List[Expression], step) -> None:
        super(LSTMState, self).__init__()
        self.hts = hts
        self.step = step

    def add_input(self, x: Expression) -> 'LSTMState':
        hts, cts = self.step(x, self.hts)
        return LSTMState(hts, self.step)

    def output(self) -> Expression:
        return self.hts[-1]


class IndRNN(nn.Module):
    def __init__(self, pc: ParameterCollection, num_layers: int,
                 input_size: int, hidden_size: int, activation=LeakyRelu(0.2),
                 W_initializer=None, u_initializer=One(), b_initializer=Zero(), h_initializer=Uniform()) -> None:
        super().__init__(pc)

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation

        _commons = dict(
            activation=activation,
            W_initializer=W_initializer,
            u_initializer=u_initializer,
            b_initializer=b_initializer,
            h_initializer=h_initializer,
        )

        self.rnn0 = IndRNNCell(self.pc, input_size, hidden_size, **_commons)
        for ix in range(1, self.num_layers):
            setattr(self, f'rnn{ix}', IndRNNCell(self.pc, hidden_size, hidden_size, **_commons))

    def init_state(self) -> 'LSTMState':
        hts = [getattr(self, f'rnn{ix}').h0.expr(moire.config.train)
               for ix in range(self.num_layers)]
        return LSTMState(hts, self.__call__)

    def __call__(self, x: Expression, htm1s: List[Expression] = None) -> List[Expression]:
        if htm1s is None:
            htm1s = itertools.repeat(None, self.num_layers)

        hs = []
        for ix, htm1 in enumerate(htm1s):
            x = getattr(self, f'rnn{ix}')(x, htm1)
            hs.append(x)

        return hs

    def transduce(self, xs: List[Expression], htm1s: List[Expression] = None) -> List[Expression]:
        assert len(xs) > 0

        hts = []
        for x in xs:
            htm1s = self.__call__(x, htm1s)
            hts.append(htm1s[-1])
        return hts

    def compress(self, xs: List[Expression], htm1s: List[Expression] = None) -> Expression:
        if len(xs) == 0:
            return self.init_state().hts[-1]
        return self.transduce(xs, htm1s)[-1]


class BiIndRNN(nn.Module):
    def __init__(self, pc: ParameterCollection, num_layers: int,
                 input_size: int, output_size: int, merge_strategy: str,
                 activation=LeakyRelu(0.2),
                 W_initializer=None, u_initializer=One(), b_initializer=Zero(), h_initializer=Uniform()) -> None:
        super(BiIndRNN, self).__init__(pc)
        assert merge_strategy in ['cat', 'avg', 'sum', 'max']

        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size

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

        _commons = dict(
            activation=activation,
            W_initializer=W_initializer,
            u_initializer=u_initializer,
            b_initializer=b_initializer,
            h_initializer=h_initializer,
        )

        self.f = IndRNN(self.pc, num_layers, input_size, forward_size, **_commons)
        self.b = IndRNN(self.pc, num_layers, input_size, backward_size, **_commons)

    def transduce(self, xs: List[Expression],
                  fhtm1s: List[Expression] = None, bhtm1s: List[Expression] = None) -> List[Expression]:
        fs = self.f.transduce(xs, fhtm1s)
        bs = self.b.transduce(xs[::-1], bhtm1s)[::-1]
        return [self.merge([f, b]) for f, b in zip(fs, bs)]

    def compress(self, xs: List[Expression],
                 fhtm1s: List[Expression] = None, bhtm1s: List[Expression] = None) -> Expression:
        if len(xs) == 0:
            f = self.f.init_state().hts[-1]
            b = self.b.init_state().hts[-1]
        else:
            f = self.f.transduce(xs, fhtm1s)[-1]
            b = self.b.transduce(xs[::-1], bhtm1s)[0]
        return self.merge([f, b])


if __name__ == '__main__':
    rnn = BiIndRNN(ParameterCollection(), 3, 4, 5, 'avg')
    dy.renew_cg()

    xs = [
        dy.inputVector([1, 2, 3, 4]),
        dy.inputVector([1, 2, 3, 4]),
        dy.inputVector([1, 2, 3, 4]),
        dy.inputVector([1, 2, 3, 4]),
    ]
    for y in rnn.transduce(xs):
        moire.debug(f'y :: {y.dim()} => {y.value()}')
    z = rnn.compress(xs)
    moire.debug(f'z :: {z.dim()} => {z.value()}')
