import dynet as dy

import moire
from moire import nn, ParameterCollection, Expression


class GRUCell(nn.Module):
    def __init__(self, pc: ParameterCollection, input_size: int, hidden_size: int,
                 activation=dy.tanh, recurrent_activation=dy.logistic) -> None:
        super(GRUCell, self).__init__(pc)

        self.h0 = self.pc.add_parameters((hidden_size,), init=dy.NormalInitializer())

        self.Wr = self.pc.add_parameters((hidden_size, input_size), init=dy.GlorotInitializer(is_lookup=False))
        self.Wz = self.pc.add_parameters((hidden_size, input_size), init=dy.GlorotInitializer(is_lookup=False))
        self.Wn = self.pc.add_parameters((hidden_size, input_size), init=dy.GlorotInitializer(is_lookup=False))

        self.Ur = self.pc.add_parameters((hidden_size, hidden_size), init=dy.SaxeInitializer())
        self.Uz = self.pc.add_parameters((hidden_size, hidden_size), init=dy.SaxeInitializer())
        self.Un = self.pc.add_parameters((hidden_size, hidden_size), init=dy.SaxeInitializer())

        self.bwr = self.pc.add_parameters((hidden_size,), init=dy.ConstInitializer(0.0))
        self.bwz = self.pc.add_parameters((hidden_size,), init=dy.ConstInitializer(0.0))
        self.bwn = self.pc.add_parameters((hidden_size,), init=dy.ConstInitializer(0.0))

        self.bur = self.pc.add_parameters((hidden_size,), init=dy.ConstInitializer(0.0))
        self.buz = self.pc.add_parameters((hidden_size,), init=dy.ConstInitializer(0.0))
        self.bun = self.pc.add_parameters((hidden_size,), init=dy.ConstInitializer(0.0))

        self.fr = recurrent_activation
        self.fz = recurrent_activation
        self.fn = activation

    def __call__(self, x: Expression, htm1: Expression = None) -> Expression:
        Wr = self.Wr.expr(moire.config.train)
        Wz = self.Wz.expr(moire.config.train)
        Wn = self.Wn.expr(moire.config.train)

        bwr = self.bwr.expr(moire.config.train)
        bwz = self.bwz.expr(moire.config.train)
        bwn = self.bwn.expr(moire.config.train)

        Ur = self.Ur.expr(moire.config.train)
        Uz = self.Uz.expr(moire.config.train)
        Un = self.Un.expr(moire.config.train)

        bur = self.bur.expr(moire.config.train)
        buz = self.buz.expr(moire.config.train)
        bun = self.bun.expr(moire.config.train)

        if htm1 is None:
            htm1 = self.h0.expr(moire.config.train)

        r = self.fr(dy.affine_transform([bwr, Wr, x]) + dy.affine_transform([bur, Ur, htm1]))
        z = self.fz(dy.affine_transform([bwz, Wz, x]) + dy.affine_transform([buz, Uz, htm1]))
        n = self.fn(dy.affine_transform([bwn, Wn, x]) + dy.cmult(r, dy.affine_transform([bun, Un, htm1])))

        ht = dy.cmult(1.0 - z, n) + dy.cmult(z, htm1)

        return ht


class GRU(nn.Module):
    pass
