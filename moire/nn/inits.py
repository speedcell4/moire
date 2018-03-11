import dynet as dy
import numpy as np
from chainer import initializers

from moire.nn.sigmoids import sigmoid
from moire.nn.thresholds import linear, relu
import warnings
from moire.nn.trigonometry import tanh

# http://pytorch.org/docs/0.3.1/nn.html#torch.nn.init.calculate_gain
activation_gains = {
    linear: 1.0,
    sigmoid: 1.0,
    tanh: 5.0 / 3.0,
    relu: float(np.sqrt(2.0)),
}


def calculate_gain(activation) -> float:
    try:
        return activation_gains[activation]
    except KeyError as _:
        try:
            return activation.gain
        except AttributeError as _:
            warnings.warn(f'calculate_gain :: unknown activation {activation}')
            return 1.0


class Initializer(object):
    def __init__(self, dtype) -> None:
        self.dtype = dtype

    def __call__(self, shape):
        raise NotImplementedError


class ChainerInitializer(Initializer):
    def __init__(self, initializer, dtype) -> None:
        super().__init__(dtype)
        self.initializer = initializer

    def __call__(self, shape):
        array = initializers.generate_array(
            self.initializer, shape=shape, xp=np)
        return dy.NumpyInitializer(array)


class Zero(ChainerInitializer):
    def __init__(self, dtype=np.float) -> None:
        super().__init__(initializers.Zero(dtype=dtype), dtype)


class One(ChainerInitializer):
    def __init__(self, dtype=np.float) -> None:
        super().__init__(initializers.One(dtype=dtype), dtype)


class Constant(ChainerInitializer):
    def __init__(self, fill_value: float, dtype=np.float) -> None:
        super().__init__(initializers.Constant(fill_value=fill_value, dtype=dtype), dtype)


class Normal(Initializer):
    def __init__(self, mean: float = 0.0, stddev: float = 0.05, dtype=np.float) -> None:
        super().__init__(dtype)
        self.mean = mean
        self.stddev = stddev

    def __call__(self, shape):
        array = np.random.normal(size=shape, loc=self.mean, scale=self.stddev).astype(self.dtype)
        return dy.NumpyInitializer(array)


class Uniform(Initializer):
    def __init__(self, minval: float = -0.05, maxval: float = 0.05, dtype=np.float32) -> None:
        super().__init__(dtype)
        self.minval = minval
        self.maxval = maxval

    def __call__(self, shape):
        array = np.random.uniform(size=shape, low=self.minval, high=self.maxval).astype(self.dtype)
        return dy.NumpyInitializer(array)


class TruncatedNormal(Initializer):
    def __init__(self, mean: float = 0.0, stddev: float = 1.0, dtype=np.float32) -> None:
        super().__init__(dtype)
        self.mean = mean
        self.stddev = stddev

    def __call__(self, shape):
        array = np.random.normal(size=shape, loc=self.mean, scale=self.stddev).astype(self.dtype)
        np.clip(array, self.mean - 2 * self.stddev, self.mean + 2 * self.stddev, array)
        return dy.NumpyInitializer(array)


# TODO VarianceScaling

class Orthogonal(ChainerInitializer):
    def __init__(self, gain: float = 1.0, dtype=np.float32) -> None:
        super().__init__(initializers.Orthogonal(scale=gain, dtype=dtype), dtype)


class Identity(ChainerInitializer):
    def __init__(self, gain: float = 1.0, dtype=np.float32) -> None:
        super(Identity, self).__init__(initializers.Identity(scale=gain, dtype=dtype), dtype)


class LecunUniform(ChainerInitializer):
    def __init__(self, gain: float = 1.0, dtype=np.float32) -> None:
        super().__init__(initializers.LeCunUniform(scale=gain, dtype=dtype), dtype)


class LecunNormal(ChainerInitializer):
    def __init__(self, gain: float = 1.0, dtype=np.float32) -> None:
        super().__init__(initializers.LeCunNormal(scale=gain, dtype=dtype), dtype)


class GlorotUniform(ChainerInitializer):
    def __init__(self, gain: float = 1.0, dtype=np.float32) -> None:
        super().__init__(initializers.GlorotUniform(scale=gain, dtype=dtype), dtype)


Xavier = GlorotUniform


class GlorotNormal(ChainerInitializer):
    def __init__(self, gain: float = 1.0, dtype=np.float32) -> None:
        super().__init__(initializers.GlorotNormal(scale=gain, dtype=dtype), dtype)


class HeUniform(ChainerInitializer):
    def __init__(self, gain: float = 1.0, dtype=np.float32) -> None:
        super().__init__(initializers.HeUniform(scale=gain, dtype=dtype), dtype)


class HeNormal(ChainerInitializer):
    def __init__(self, gain: float = 1.0, dtype=np.float32) -> None:
        super().__init__(initializers.HeNormal(scale=gain, dtype=dtype), dtype)


if __name__ == '__main__':
    from moire import Expression, ParameterCollection

    pc: dy.Model = ParameterCollection()
    w = pc.add_parameters((10, 10), GlorotNormal()((10, 10)))

    print(w.as_array())
