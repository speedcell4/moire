import unittest

import dynet as dy

from moire import ParameterCollection, normal
from moire.nn import Linear, MLP, BiLinear


class TestLinear(unittest.TestCase):
    def test_shape_bias(self, in_features=3, out_features=4):
        pc = ParameterCollection()
        linear = Linear(pc, in_features, out_features)

        dy.renew_cg()
        x = normal(in_features)
        y = linear(x)
        self.assertTrue(y.dim(), ((out_features,), 1))

    def test_shape_nobias(self, in_features=3, out_features=4):
        model = Linear(ParameterCollection(), in_features, out_features, bias=False)

        dy.renew_cg()
        x = normal(in_features)
        self.assertTrue(model(x).dim(), ((out_features,), 1))


class TestMLP(unittest.TestCase):
    def test_shape_multi_layers(self, num_layers=5, in_features=3, out_features=4):
        model = MLP(ParameterCollection(), num_layers, in_features, out_features)

        dy.renew_cg()
        x = normal(in_features)
        self.assertTrue(model(x).dim(), ((out_features,), 1))

    def test_shape_single_layers(self, num_layers=1, in_features=3, out_features=4):
        model = MLP(ParameterCollection(), num_layers, in_features, out_features)

        dy.renew_cg()
        x = normal(in_features)
        self.assertTrue(model(x).dim(), ((out_features,), 1))


class TestBiLinear(unittest.TestCase):
    def test_shape(self, in1_features=3, in2_features=5, out_features=7):
        model = BiLinear(ParameterCollection(), in1_features, in2_features, out_features)

        dy.renew_cg()
        x1 = normal(in1_features)
        x2 = normal(in2_features)
        self.assertTrue(model(x1, x2).dim(), ((out_features,), 1))