import warnings

import dynet as dy
import numpy as np
import chainer

from moire.nn.functions import sigmoid, linear, relu, tanh

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
        return dy.NumpyInitializer(self.generate_array(shape))

    def generate_array(self, shape):
        raise NotImplementedError


class NumpyInitializer(Initializer):
    def __init__(self, array, dtype) -> None:
        super().__init__(dtype)
        self.array = array

    def generate_array(self, shape):
        assert self.array.shape == shape
        return self.array


class ChainerInitializer(Initializer):
    def __init__(self, initializer, dtype) -> None:
        super().__init__(dtype)
        self.initializer = initializer

    def generate_array(self, shape):
        return chainer.initializers.generate_array(
            self.initializer, shape=shape, xp=np)


class ConcatenatedInitializer(Initializer):
    def __init__(self, *initializers, axis: int) -> None:
        assert all(init.dtype == initializers[0].dtype for init in initializers)
        super().__init__(initializers[0].dtype)

        self.axis = axis
        self.initializers = initializers
        self.nb_initializers = len(initializers)

    def generate_array(self, shape):
        shape = list(shape)
        shape[self.axis] //= self.nb_initializers
        return np.concatenate(
            [init.generate_array(shape) for init in self.initializers], axis=self.axis)


class Zero(ChainerInitializer):
    def __init__(self, dtype=np.float) -> None:
        super().__init__(chainer.initializers.Zero(dtype=dtype), dtype)


class One(ChainerInitializer):
    def __init__(self, dtype=np.float) -> None:
        super().__init__(chainer.initializers.One(dtype=dtype), dtype)


class Constant(ChainerInitializer):
    def __init__(self, fill_value: float, dtype=np.float) -> None:
        super().__init__(chainer.initializers.Constant(fill_value=fill_value, dtype=dtype), dtype)


class Normal(Initializer):
    def __init__(self, mean: float = 0.0, stddev: float = 0.05, dtype=np.float) -> None:
        super().__init__(dtype)
        self.mean = mean
        self.stddev = stddev

    def generate_array(self, shape):
        return np.random.normal(size=shape, loc=self.mean, scale=self.stddev).astype(self.dtype)


class Uniform(Initializer):
    def __init__(self, minval: float = -0.05, maxval: float = 0.05, dtype=np.float32) -> None:
        super().__init__(dtype)
        self.minval = minval
        self.maxval = maxval

    def generate_array(self, shape):
        return np.random.uniform(size=shape, low=self.minval, high=self.maxval).astype(self.dtype)


class TruncatedNormal(Initializer):
    def __init__(self, mean: float = 0.0, stddev: float = 1.0, dtype=np.float32) -> None:
        super().__init__(dtype)
        self.mean = mean
        self.stddev = stddev

    def generate_array(self, shape):
        array = np.random.normal(size=shape, loc=self.mean, scale=self.stddev).astype(self.dtype)
        np.clip(array, self.mean - 2 * self.stddev, self.mean + 2 * self.stddev, array)
        return array


# TODO VarianceScaling

class Orthogonal(ChainerInitializer):
    def __init__(self, gain: float = 1.0, dtype=np.float32) -> None:
        super().__init__(chainer.initializers.Orthogonal(scale=gain, dtype=dtype), dtype)


class Identity(ChainerInitializer):
    def __init__(self, gain: float = 1.0, dtype=np.float32) -> None:
        super(Identity, self).__init__(chainer.initializers.Identity(scale=gain, dtype=dtype), dtype)


class LecunUniform(ChainerInitializer):
    def __init__(self, gain: float = 1.0, dtype=np.float32) -> None:
        super().__init__(chainer.initializers.LeCunUniform(scale=gain, dtype=dtype), dtype)


class LecunNormal(ChainerInitializer):
    def __init__(self, gain: float = 1.0, dtype=np.float32) -> None:
        super().__init__(chainer.initializers.LeCunNormal(scale=gain, dtype=dtype), dtype)


class GlorotUniform(ChainerInitializer):
    def __init__(self, gain: float = 1.0, dtype=np.float32) -> None:
        super().__init__(chainer.initializers.GlorotUniform(scale=gain, dtype=dtype), dtype)


Xavier = GlorotUniform


class GlorotNormal(ChainerInitializer):
    def __init__(self, gain: float = 1.0, dtype=np.float32) -> None:
        super().__init__(chainer.initializers.GlorotNormal(scale=gain, dtype=dtype), dtype)


class HeUniform(ChainerInitializer):
    def __init__(self, gain: float = 1.0, dtype=np.float32) -> None:
        super().__init__(chainer.initializers.HeUniform(scale=gain, dtype=dtype), dtype)


class HeNormal(ChainerInitializer):
    def __init__(self, gain: float = 1.0, dtype=np.float32) -> None:
        super().__init__(chainer.initializers.HeNormal(scale=gain, dtype=dtype), dtype)


class Positional(Initializer):
    def __init__(self, dtype):
        super().__init__(dtype)

    def generate_array(self, shape):
        (max_len, model_size) = shape
        p_e = np.arange(model_size) // 2 * 2 / model_size
