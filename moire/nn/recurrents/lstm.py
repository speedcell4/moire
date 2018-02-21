import dynet as dy

from moire import nn, Expression, ParameterCollection


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

    def __call__(self, x: Expression, htm1: Expression = None, ctm1: Expression = None):
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
