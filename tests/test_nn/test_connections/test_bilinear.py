import unittest

import dynet as dy
import numpy as np

from moire import ParameterCollection
from moire.nn import BiLinear


class TestBiLinear(unittest.TestCase):
    def _test_shape(self, in1_feature, in2_feature, out_feature, use_u, use_v, use_bias):
        fc = BiLinear(ParameterCollection(), in1_feature, in2_feature, out_feature, use_v=use_v, use_u=use_u,
                      use_bias=use_bias)
        dy.renew_cg()

        x1 = dy.inputVector(np.random.random(in1_feature, ))
        x2 = dy.inputVector(np.random.random(in2_feature, ))
        y = fc.__call__(x1, x2)
        self.assertEqual(y.dim(), ((out_feature,), 1))

    def test_shape(self):
        for use_u in [True, False]:
            for use_v in [True, False]:
                for use_bias in [True, False]:
                    self._test_shape(3, 4, 5, use_u, use_v, use_bias)
