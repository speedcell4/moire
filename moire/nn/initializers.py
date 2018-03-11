import dynet as dy
import numpy as np
from chainer import initializers
from moire.nn.trigonometry import tanh
from moire.nn.thresholds import linear, relu
from moire.nn.sigmoids import sigmoid


# http://pytorch.org/docs/0.3.1/nn.html#torch.nn.init.calculate_gain
def calculate_gain(activation) -> float:
    try:
        return {
            linear: 1.0,
            sigmoid: 1.0,
            tanh: 5.0 / 3.0,
            relu: float(np.sqrt(2.0)),
        }[activation]
    except KeyError as _:
        try:
            return activation.gain
        except AttributeError as _:
            return 1.0


class Initializer(object):
    def __init__(self, shape, dtype) -> None:
        self.shape = shape
        self.dtype = dtype

    def __call__(self):
        raise NotImplementedError


class ChainerInitializer(Initializer):
    def __init__(self, initializer, shape, dtype) -> None:
        super().__init__(shape, dtype)
        self.initializer = initializer

    def __call__(self):
        array = initializers.generate_array(
            self.initializer, shape=self.shape, xp=np)
        return dy.NumpyInitializer(array)


class Zeros(ChainerInitializer):
    def __init__(self, shape, dtype=np.float) -> None:
        super().__init__(initializers.Zero(dtype=dtype), shape, dtype)


class Ones(ChainerInitializer):
    def __init__(self, shape, dtype=np.float) -> None:
        super().__init__(initializers.One(dtype=dtype), shape, dtype)


class Constant(ChainerInitializer):
    def __init__(self, shape, fill_value: float, dtype=np.float) -> None:
        super().__init__(initializers.Constant(fill_value=fill_value, dtype=dtype), shape, dtype)


class Normal(Initializer):
    def __init__(self, shape, mean: float = 0.0, stddev: float = 0.05, dtype=np.float) -> None:
        super().__init__(shape, dtype)
        self.mean = mean
        self.stddev = stddev

    def __call__(self):
        array = np.random.normal(size=self.shape, loc=self.mean, scale=self.stddev).astype(self.dtype)
        return dy.NumpyInitializer(array)


class Uniform(Initializer):
    def __init__(self, shape, minval: float = -0.05, maxval: float = 0.05, dtype=np.float32) -> None:
        super().__init__(shape, dtype)
        self.minval = minval
        self.maxval = maxval

    def __call__(self):
        array = np.random.uniform(size=self.shape, low=self.minval, high=self.maxval).astype(self.dtype)
        return dy.NumpyInitializer(array)


class TruncatedNormal(Initializer):
    def __init__(self, shape, mean: float = 0.0, stddev: float = 1.0, dtype=np.float32) -> None:
        super().__init__(shape, dtype)
        self.mean = mean
        self.stddev = stddev

    def __call__(self):
        array = np.random.normal(size=self.shape, loc=self.mean, scale=self.stddev).astype(self.dtype)
        np.clip(array, self.mean - 2 * self.stddev, self.mean + 2 * self.stddev, array)
        return dy.NumpyInitializer(array)


# TODO VarianceScaling

class Orthogonal(ChainerInitializer):
    def __init__(self, shape, gain: float = 1.0, dtype=np.float32) -> None:
        super().__init__(initializers.Orthogonal(scale=gain, dtype=dtype), shape, dtype)


class Identity(ChainerInitializer):
    def __init__(self, shape, gain: float = 1.0, dtype=np.float32) -> None:
        super(Identity, self).__init__(initializers.Identity(scale=gain, dtype=dtype), shape, dtype)


class LecunUniform(ChainerInitializer):
    def __init__(self, shape, gain: float = 1.0, dtype=np.float32) -> None:
        super().__init__(initializers.LeCunUniform(scale=gain, dtype=dtype), shape, dtype)


class LecunNormal(ChainerInitializer):
    def __init__(self, shape, gain: float = 1.0, dtype=np.float32) -> None:
        super().__init__(initializers.LeCunNormal(scale=gain, dtype=dtype), shape, dtype)


class GlorotUniform(ChainerInitializer):
    def __init__(self, shape, gain: float = 1.0, dtype=np.float32) -> None:
        super().__init__(initializers.GlorotUniform(scale=gain, dtype=dtype), shape, dtype)


Xavier = GlorotUniform


class GlorotNormal(ChainerInitializer):
    def __init__(self, shape, gain: float = 1.0, dtype=np.float32) -> None:
        super().__init__(initializers.GlorotNormal(scale=gain, dtype=dtype), shape, dtype)


class HeUniform(ChainerInitializer):
    def __init__(self, shape, gain: float = 1.0, dtype=np.float32) -> None:
        super().__init__(initializers.HeUniform(scale=gain, dtype=dtype), shape, dtype)


class HeNormal(ChainerInitializer):
    def __init__(self, shape, gain: float = 1.0, dtype=np.float32) -> None:
        super().__init__(initializers.HeNormal(scale=gain, dtype=dtype), shape, dtype)


if __name__ == '__main__':
    from moire import Expression, ParameterCollection
    from moire.nn.thresholds import LeakyRelu

    pc: dy.Model = ParameterCollection()
    w = pc.add_parameters((10, 10), Ones(shape=(10, 10))())
    leaky_relu = LeakyRelu(0.2)

    print(calculate_gain(leaky_relu))
    print(w.as_array())
