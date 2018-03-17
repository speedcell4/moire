import itertools
from typing import List, Tuple

import dynet as dy

import moire
from moire import Expression, ParameterCollection, nn
from moire.nn.initializers import ConcatenatedInitializer, GlorotNormal, One, Orthogonal, Uniform, Zero
from moire.nn.sigmoids import sigmoid
from moire.nn.trigonometry import tanh


class LSTMCell(nn.Module):
    G_I: int = 0
    G_F: int = 1
    G_G: int = 2
    G_O: int = 3

    def __init__(self, pc: ParameterCollection,
                 input_size: int, hidden_size: int,
                 dropout_ratio: float = None, zoneout_ratio: float = None,
                 activation=tanh, recurrent_activation=sigmoid,
                 kernel_initializer=GlorotNormal(), recurrent_initializer=Orthogonal(),
                 hidden_initializer=Uniform(), bias_initializer=Zero(), forget_bias_initializer=One()) -> None:
        super(LSTMCell, self).__init__(pc)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_ratio = dropout_ratio
        self.zoneout_ratio = zoneout_ratio

        self.h0 = self.add_param((hidden_size,), hidden_initializer)
        self.c0 = self.add_param((hidden_size,), hidden_initializer)

        kernel_initializer = ConcatenatedInitializer(
            kernel_initializer, kernel_initializer, kernel_initializer, kernel_initializer, axis=0)
        self.W = self.add_param((4 * hidden_size, input_size), kernel_initializer)

        recurrent_initializer = ConcatenatedInitializer(
            recurrent_initializer, recurrent_initializer, recurrent_initializer, recurrent_initializer, axis=0)
        self.U = self.add_param((4 * hidden_size, hidden_size), recurrent_initializer)

        bias_initializer = ConcatenatedInitializer(
            bias_initializer, forget_bias_initializer, bias_initializer, bias_initializer, axis=0)
        self.b = self.add_param((4 * hidden_size,), bias_initializer)

        self.fi = recurrent_activation
        self.ff = recurrent_activation
        self.fg = activation
        self.fo = recurrent_activation

        self.dropout = nn.Dropout(dropout_ratio)
        self.zoneout = nn.Zoneout(zoneout_ratio)

    def __call__(self, x: Expression,
                 htm1: Expression = None, ctm1: Expression = None) -> Tuple[Expression, Expression]:
        W = self.W.expr(moire.config.train)
        U = self.U.expr(moire.config.train)
        b = self.b.expr(moire.config.train)

        x = self.dropout(x)

        if htm1 is None:
            htm1 = self.h0.expr(moire.config.train)
        if ctm1 is None:
            ctm1 = self.c0.expr(moire.config.train)

        y = dy.reshape(dy.affine_transform([b, W, x, U, ctm1]), (4, self.hidden_size))
        i = self.fi(y[self.G_I])
        f = self.ff(y[self.G_F])
        g = self.fg(y[self.G_G])
        o = self.fo(y[self.G_O])

        ct = dy.cmult(f, ctm1) + dy.cmult(i, g)
        ht = dy.cmult(o, self.fg(ctm1))

        ct = self.zoneout(ct, ctm1)
        ht = self.zoneout(ht, htm1)

        return ht, ct


class LSTMState(object):
    __slots__ = ('hts', 'cts', 'step')

    def __init__(self, hts: List[Expression], cts: List[Expression], step) -> None:
        super(LSTMState, self).__init__()
        self.hts = hts
        self.cts = cts
        self.step = step

    def add_input(self, x: Expression) -> 'LSTMState':
        hts, cts = self.step(x, self.hts, self.cts)
        return LSTMState(hts, cts, self.step)

    def output(self) -> Expression:
        return self.hts[-1]


class LSTM(nn.Module):
    def __init__(self, pc: ParameterCollection, num_layers: int,
                 input_size: int, hidden_size: int,
                 dropout_ratio: float = None, zoneout_ratio: float = None,
                 activation=tanh, recurrent_activation=sigmoid,
                 kernel_initializer=GlorotNormal(), recurrent_initializer=Orthogonal(),
                 hidden_initializer=Uniform(), bias_initializer=Zero(), forget_bias_initializer=One()) -> None:
        super(LSTM, self).__init__(pc)

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        _commons = dict(
            dropout_ratio=dropout_ratio, zoneout_ratio=zoneout_ratio,
            activation=activation, recurrent_activation=recurrent_activation,
            kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer,
            hidden_initializer=hidden_initializer,
            bias_initializer=bias_initializer, forget_bias_initializer=forget_bias_initializer,
        )

        self.rnn0 = LSTMCell(self.pc, input_size, hidden_size, **_commons)
        for ix in range(1, num_layers):
            setattr(self, f'rnn{ix}', LSTMCell(self.pc, hidden_size, hidden_size, **_commons))

    def init_state(self) -> 'LSTMState':
        hts = [getattr(self, f'rnn{ix}').h0.expr(moire.config.train)
               for ix in range(self.num_layers)]
        cts = [getattr(self, f'rnn{ix}').c0.expr(moire.config.train)
               for ix in range(self.num_layers)]
        return LSTMState(hts, cts, self.__call__)

    def __call__(self, x: Expression, htm1s: List[Expression] = None, ctm1s: List[Expression] = None):
        hts, cts = [], []

        if htm1s is None:
            htm1s = itertools.repeat(None, self.num_layers)
        if ctm1s is None:
            ctm1s = itertools.repeat(None, self.num_layers)
        for ix, (htm1, ctm1) in enumerate(zip(htm1s, ctm1s)):
            ht, ct = getattr(self, f'rnn{ix}')(x, htm1, ctm1)
            hts.append(ht)
            cts.append(ct)
            x = ht

        return hts, cts

    def transduce(self, xs: List[Expression],
                  htm1s: List[Expression] = None, ctm1s: List[Expression] = None) -> List[Expression]:
        assert len(xs) > 0

        hts = []
        for x in xs:
            htm1s, ctm1s = self.__call__(x, htm1s, ctm1s)
            hts.append(htm1s[-1])
        return hts

    def compress(self, xs: List[Expression],
                 htm1s: List[Expression] = None, ctm1s: List[Expression] = None) -> Expression:
        if len(xs) == 0:
            return self.init_state().hts[-1]
        return self.transduce(xs, htm1s, ctm1s)[-1]


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
        fs = self.f.transduce(xs, fhtm1s, fctm1s)
        bs = self.b.transduce(xs[::-1], bhtm1s, bctm1s)[::-1]
        return [self.merge([f, b]) for f, b in zip(fs, bs)]

    def compress(self, xs: List[Expression],
                 fhtm1s: List[Expression] = None, fctm1s: List[Expression] = None,
                 bhtm1s: List[Expression] = None, bctm1s: List[Expression] = None) -> Expression:
        if len(xs) == 0:
            f = self.f.init_state().hts[-1]
            b = self.b.init_state().hts[-1]
        else:
            f = self.f.transduce(xs, fhtm1s, fctm1s)[-1]
            b = self.b.transduce(xs[::-1], bhtm1s, bctm1s)[0]
        return self.merge([f, b])


if __name__ == '__main__':
    cell = LSTMCell(ParameterCollection(), 4, 5)

    dy.renew_cg()

    x = dy.inputVector([1, 2, 3, 4])
    h, c = cell.__call__(x)

    print(f'h :: {h.dim()} => {h.value()}')
    print(f'c :: {c.dim()} => {c.value()}')
