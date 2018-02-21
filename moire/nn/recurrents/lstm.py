import itertools
from typing import Tuple, List

import dynet as dy

from moire import nn, Expression, ParameterCollection
from moire.nn.recurrents.utils import scan

__all__ = [
    'LSTMState', 'LSTMCell', 'LSTM', 'BiLSTM',
]


class LSTMCell(nn.Module):
    def __init__(self, pc: ParameterCollection,
                 input_size: int, hidden_size: int,
                 dropout: float = 0.05, zoneout: float = 0.05,
                 activation=dy.tanh, recurrent_activation=dy.logistic) -> None:
        super(LSTMCell, self).__init__(pc)

        self.h0 = self.pc.add_parameters((hidden_size,), init=dy.NormalInitializer())
        self.c0 = self.pc.add_parameters((hidden_size,), init=dy.NormalInitializer())

        self.Wi = self.pc.add_parameters((hidden_size, input_size), init=dy.GlorotInitializer(is_lookup=False))
        self.Wf = self.pc.add_parameters((hidden_size, input_size), init=dy.GlorotInitializer(is_lookup=False))
        self.Wg = self.pc.add_parameters((hidden_size, input_size), init=dy.GlorotInitializer(is_lookup=False))
        self.Wo = self.pc.add_parameters((hidden_size, input_size), init=dy.GlorotInitializer(is_lookup=False))

        self.Ui = self.pc.add_parameters((hidden_size, hidden_size), init=dy.SaxeInitializer())
        self.Uf = self.pc.add_parameters((hidden_size, hidden_size), init=dy.SaxeInitializer())
        self.Ug = self.pc.add_parameters((hidden_size, hidden_size), init=dy.SaxeInitializer())
        self.Uo = self.pc.add_parameters((hidden_size, hidden_size), init=dy.SaxeInitializer())

        self.bi = self.pc.add_parameters((hidden_size,), init=dy.ConstInitializer(0.0))
        self.bf = self.pc.add_parameters((hidden_size,), init=dy.ConstInitializer(1.0))
        self.bg = self.pc.add_parameters((hidden_size,), init=dy.ConstInitializer(0.0))
        self.bo = self.pc.add_parameters((hidden_size,), init=dy.ConstInitializer(0.0))

        self.fi = recurrent_activation
        self.ff = recurrent_activation
        self.fg = activation
        self.fo = recurrent_activation

        self.dropout_ratio = dropout
        self.zoneout_ratio = zoneout

        self.dropout = nn.Dropout(dropout)
        self.zoneout = nn.Zoneout(zoneout)

    def __call__(self, x: Expression, htm1: Expression = None, ctm1: Expression = None) \
            -> Tuple[Expression, Expression]:
        Wi = self.Wi.expr(self.training)
        Wf = self.Wf.expr(self.training)
        Wg = self.Wg.expr(self.training)
        Wo = self.Wo.expr(self.training)

        Ui = self.Ui.expr(self.training)
        Uf = self.Uf.expr(self.training)
        Ug = self.Ug.expr(self.training)
        Uo = self.Uo.expr(self.training)

        bi = self.bi.expr(self.training)
        bf = self.bf.expr(self.training)
        bg = self.bg.expr(self.training)
        bo = self.bo.expr(self.training)

        x = self.dropout(x)

        if htm1 is None:
            htm1 = self.h0.expr(self.training)
        if ctm1 is None:
            ctm1 = self.c0.expr(self.training)

        i = self.fi(dy.affine_transform([bi, Wi, x, Ui, htm1]))
        f = self.ff(dy.affine_transform([bf, Wf, x, Uf, htm1]))
        g = self.fg(dy.affine_transform([bg, Wg, x, Ug, htm1]))
        o = self.fo(dy.affine_transform([bo, Wo, x, Uo, htm1]))

        ct = dy.cmult(f, ctm1) + dy.cmult(i, g)
        ct = self.zoneout(ct, ctm1)

        ht = dy.cmult(o, self.fg(ctm1))
        ht = self.zoneout(ht, htm1)

        return ht, ct


class LSTMState(object):
    def __init__(self, hts: List[Expression], cts: List[Expression], apply):
        self.hts = hts
        self.cts = cts
        self.apply = apply

    def add_input(self, x: Expression) -> 'LSTMState':
        hts, cts = self.apply(x, self.hts, self.cts)
        return LSTMState(hts, cts, self.apply)

    def output(self):
        return self.hts[-1]


class LSTM(nn.Module):
    def __init__(self, pc: ParameterCollection, num_layers: int,
                 input_size: int, hidden_size: int,
                 dropout: float = 0.05, zoneout: float = 0.05,
                 activation=dy.tanh, recurrent_activation=dy.logistic):
        super(LSTM, self).__init__(pc)

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        _commons = dict(
            dropout=dropout, zoneout=zoneout,
            activation=activation, recurrent_activation=recurrent_activation,
        )

        self.rnn0 = LSTMCell(self.pc, input_size, hidden_size, **_commons)
        for ix in range(1, num_layers):
            setattr(self, f'rnn{ix}', LSTMCell(self.pc, hidden_size, hidden_size, **_commons))

    def init_state(self) -> 'LSTMState':
        hts = [getattr(self, f'rnn{ix}').h0.expr(self.training) for ix in range(self.num_layers)]
        cts = [getattr(self, f'rnn{ix}').c0.expr(self.training) for ix in range(self.num_layers)]
        return LSTMState(hts, cts, self.__call__)

    def __call__(self, x: Expression, htm1s: List[Expression] = None, ctm1s: List[Expression] = None):
        hts, cts = [], []

        if htm1s is None:
            htm1s = itertools.repeat(None)
        if ctm1s is None:
            ctm1s = itertools.repeat(None)
        for ix, (htm1, ctm1) in enumerate(zip(htm1s, ctm1s)):
            ht, ct = getattr(self, f'rnn{ix}')(x, htm1, ctm1)
            hts.append(ht)
            cts.append(ct)

        return hts, cts

    def transduce(self, xs: List[Expression],
                  htm1s: List[Expression] = None, ctm1s: List[Expression] = None) -> List[Expression]:
        assert len(xs) > 0

        hs, _ = zip(*scan(self.__call__, xs, htm1s, ctm1s))
        return hs

    def compress(self, xs: List[Expression],
                 htm1s: List[Expression] = None, ctm1s: List[Expression] = None) -> Expression:
        return self.transduce(xs, htm1s, ctm1s)[-1]


class BiLSTM(nn.Module):
    def __init__(self, pc: ParameterCollection, num_layers: int,
                 input_size: int, hidden_size: int,
                 dropout: float = 0.05, zoneout: float = 0.05,
                 activation=dy.tanh, recurrent_activation=dy.logistic):
        super(BiLSTM, self).__init__(pc)

        self.f = LSTM(self.pc, num_layers, input_size, hidden_size,
                      dropout, zoneout, activation, recurrent_activation)
        self.b = LSTM(self.pc, num_layers, input_size, hidden_size,
                      dropout, zoneout, activation, recurrent_activation)

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
