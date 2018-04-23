import cProfile

import dynet as dy
import numpy as np

from moire import nn
from moire import ParameterCollection

VEC_SIZE = 200
NB_SAMPLES = 10000
TIMES = 10


def generate_data():
    xs, restricts = [], []
    for _ in range(NB_SAMPLES):
        x = np.random.standard_normal((VEC_SIZE,))
        restrict = np.random.random_integers(
            low=0, high=VEC_SIZE - 1,
            size=(np.random.random_integers(low=1, high=VEC_SIZE))).tolist()

        xs.append(x)
        restricts.append(restrict)

    return xs, restricts


xs, restricts = generate_data()
p

def test_to_device():
    profile = cProfile.Profile()

    for _ in range(TIMES):
        for x, restrict in zip(xs, restricts):
            dy.renew_cg()
            x = dy.inputVector(x)
            profile.enable()
            dy.log_softmax(x, restrict)
            profile.disable()

    profile.print_stats()


def test_log_softmax():
    imp1 = cProfile.Profile()
    imp2 = cProfile.Profile()

    for _ in range(TIMES):
        for x, restrict in zip(xs, restricts):
            log_softmax = nn.LogSoftmax(ParameterCollection(), VEC_SIZE, restrict)
            dy.renew_cg()
            x = dy.inputVector(x)
            imp1.enable()
            y = log_softmax.__call__(x)
            imp1.disable()
            # z = dy.log_softmax(x, restrict)
            print(f'y :: {y.dim()} => {y.value()}')
            # print(f'z :: {z.dim()} => {z.value()}')
            # exit((1))

    imp1.print_stats()


test_log_softmax()
test_to_device()
