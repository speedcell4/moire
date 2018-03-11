import itertools
from typing import Tuple, List

import dynet as dy

import moire
from moire import nn, Expression, ParameterCollection
from moire.nn.inits import Uniform, GlorotNormal, Orthogonal, Zero, One
from moire.nn.sigmoids import sigmoid
from moire.nn.trigonometry import tanh


# TODO move to hard_sigmoid ?

class LSTMCell(nn.Module):
    def __init__(self, pc: ParameterCollection,
                 input_size: int, hidden_size: int,
                 dropout_ratio: float = None, zoneout_ratio: float = None,
                 activation=tanh, recurrent_activation=sigmoid,
                 kernel_initializer=GlorotNormal(), recurrent_initializer=Orthogonal(),
                 hidden_initializer=Uniform(), bias_initializer=Zero(), forget_bias_initializer=One()) -> None:
        super(LSTMCell, self).__init__(pc)

        self.h0 = self.add_param((hidden_size,), hidden_initializer)
        self.c0 = self.add_param((hidden_size,), hidden_initializer)

        self.Wi = self.add_param((hidden_size, input_size), kernel_initializer)
        self.Wf = self.add_param((hidden_size, input_size), kernel_initializer)
        self.Wg = self.add_param((hidden_size, input_size), kernel_initializer)
        self.Wo = self.add_param((hidden_size, input_size), kernel_initializer)

        self.Ui = self.add_param((hidden_size, hidden_size), recurrent_initializer)
        self.Uf = self.add_param((hidden_size, hidden_size), recurrent_initializer)
        self.Ug = self.add_param((hidden_size, hidden_size), recurrent_initializer)
        self.Uo = self.add_param((hidden_size, hidden_size), recurrent_initializer)

        self.bi = self.add_param((hidden_size,), bias_initializer)
        self.bf = self.add_param((hidden_size,), forget_bias_initializer)
        self.bg = self.add_param((hidden_size,), bias_initializer)
        self.bo = self.add_param((hidden_size,), bias_initializer)

        self.fi = recurrent_activation
        self.ff = recurrent_activation
        self.fg = activation
        self.fo = recurrent_activation

        self.dropout_ratio = dropout_ratio
        self.zoneout_ratio = zoneout_ratio

        self.dropout = nn.Dropout(dropout_ratio)
        self.zoneout = nn.Zoneout(zoneout_ratio)

    def __call__(self, x: Expression,
                 htm1: Expression = None, ctm1: Expression = None) -> Tuple[Expression, Expression]:
        Wi = self.Wi.expr(moire.config.train)
        Wf = self.Wf.expr(moire.config.train)
        Wg = self.Wg.expr(moire.config.train)
        Wo = self.Wo.expr(moire.config.train)

        Ui = self.Ui.expr(moire.config.train)
        Uf = self.Uf.expr(moire.config.train)
        Ug = self.Ug.expr(moire.config.train)
        Uo = self.Uo.expr(moire.config.train)

        bi = self.bi.expr(moire.config.train)
        bf = self.bf.expr(moire.config.train)
        bg = self.bg.expr(moire.config.train)
        bo = self.bo.expr(moire.config.train)

        x = self.dropout(x)

        if htm1 is None:
            htm1 = self.h0.expr(moire.config.train)
        if ctm1 is None:
            ctm1 = self.c0.expr(moire.config.train)

        i = self.fi(dy.affine_transform([bi, Wi, x, Ui, htm1]))
        f = self.ff(dy.affine_transform([bf, Wf, x, Uf, htm1]))
        g = self.fg(dy.affine_transform([bg, Wg, x, Ug, htm1]))
        o = self.fo(dy.affine_transform([bo, Wo, x, Uo, htm1]))

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
                 input_size: int, hidden_size: int,
                 dropout_ratio: float = None, zoneout_ratio: float = None,
                 activation=tanh, recurrent_activation=sigmoid,
                 kernel_initializer=GlorotNormal(), recurrent_initializer=Orthogonal(),
                 hidden_initializer=Uniform(), bias_initializer=Zero(), forget_bias_initializer=One()) -> None:
        super(BiLSTM, self).__init__(pc)

        self.f = LSTM(self.pc, num_layers, input_size, hidden_size,
                      dropout_ratio, zoneout_ratio, activation, recurrent_activation,
                      kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer,
                      hidden_initializer=hidden_initializer,
                      bias_initializer=bias_initializer, forget_bias_initializer=forget_bias_initializer)
        self.b = LSTM(self.pc, num_layers, input_size, hidden_size,
                      dropout_ratio, zoneout_ratio, activation, recurrent_activation,
                      kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer,
                      hidden_initializer=hidden_initializer,
                      bias_initializer=bias_initializer, forget_bias_initializer=forget_bias_initializer)

    def transduce(self, xs: List[Expression],
                  fhtm1s: List[Expression] = None, fctm1s: List[Expression] = None,
                  bhtm1s: List[Expression] = None, bctm1s: List[Expression] = None) -> List[Expression]:
        fs = self.f.transduce(xs, fhtm1s, fctm1s)
        bs = self.b.transduce(xs[::-1], bhtm1s, bctm1s)[::-1]
        return [dy.concatenate([f, b]) for f, b in zip(fs, bs)]

    def compress(self, xs: List[Expression],
                 fhtm1s: List[Expression] = None, fctm1s: List[Expression] = None,
                 bhtm1s: List[Expression] = None, bctm1s: List[Expression] = None) -> Expression:
        f = self.f.compress(xs, fhtm1s, fctm1s)
        b = self.b.compress(xs[::-1], bhtm1s, bctm1s)
        return dy.concatenate([f, b])


if __name__ == '__main__':
    rnn = LSTM(ParameterCollection(), 1, 4, 5)
    dy.renew_cg()

    xs = [
        dy.inputVector([1, 2, 3, 4]),
        dy.inputVector([1, 2, 3, 4]),
    ]

    s0 = rnn.init_state()
    print(s0.output().value())

    s1 = s0.add_input(dy.inputVector([1, 2, 3, 4]))
    print(s1.output().value())

    s2 = s1.add_input(dy.inputVector([1, 2, 3, 4]))
    print(s2.output().value())
