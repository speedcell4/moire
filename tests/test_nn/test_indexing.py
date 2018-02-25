import unittest

import dynet as dy
import numpy as np

from moire.nn.indexing import EpsilonArgMax, EpsilonArgMin


class TestEpsilonArgMax(unittest.TestCase):
    def test_training_mode(self):
        x = dy.inputVector([1, 2, 3])
        epsilon_argmax = EpsilonArgMax(0.5)
        epsilon_argmax.training = True
        y = [0, 0, 0]
        for _ in range(100000):
            y[epsilon_argmax(x)] += 1
        y = np.array(y) / sum(y)
        z = np.array([1 / 6, 1 / 6, 4 / 6], dtype=np.float32)
        self.assertTrue(np.allclose(y, z, atol=1.e-2))

    def test_evaluate_mode(self):
        x = dy.inputVector([1, 2, 3])
        epsilon_argmax = EpsilonArgMax(0.5)
        epsilon_argmax.training = False
        y = [0, 0, 0]
        for _ in range(100000):
            y[epsilon_argmax(x)] += 1
        y = np.array(y) / sum(y)
        z = np.array([0, 0, 1], dtype=np.float32)
        self.assertTrue(np.allclose(y, z, atol=1.e-2))


class TestEpsilonArgMin(unittest.TestCase):
    def test_training_mode(self):
        x = dy.inputVector([1, 2, 3])
        epsilon_argmin = EpsilonArgMin(0.5)
        epsilon_argmin.training = True
        y = [0, 0, 0]
        for _ in range(100000):
            y[epsilon_argmin(x)] += 1
        y = np.array(y) / sum(y)
        z = np.array([4 / 6, 1 / 6, 1 / 6], dtype=np.float32)
        self.assertTrue(np.allclose(y, z, atol=1.e-2))

    def test_evaluate_mode(self):
        x = dy.inputVector([1, 2, 3])
        epsilon_argmin = EpsilonArgMin(0.5)
        epsilon_argmin.training = False
        y = [0, 0, 0]
        for _ in range(100000):
            y[epsilon_argmin(x)] += 1
        y = np.array(y) / sum(y)
        z = np.array([1, 0, 0], dtype=np.float32)
        self.assertTrue(np.allclose(y, z, atol=1.e-2))
