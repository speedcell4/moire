import dynet as dy
import numpy as np
from chainer import initializers


def zeros(shape, dtype=np.float32):
    array = initializers.generate_array(initializers.Zero(dtype=dtype), shape=shape, xp=np)
    return dy.NumpyInitializer(array)


def ones(shape, dtype=np.float32):
    array = initializers.generate_array(initializers.One(dtype=dtype), shape=shape, xp=np)
    return dy.NumpyInitializer(array)


def constant(shape, fill_value: float, dtype=np.float32):
    array = initializers.generate_array(initializers.Constant(fill_value=fill_value, dtype=dtype), shape=shape, xp=np)
    return dy.NumpyInitializer(array)


def random_normal(shape, mean: float = 0.0, stddev: float = 0.05, dtype=np.float32):
    array = np.random.normal(size=shape, loc=mean, scale=stddev).astype(dtype)
    return dy.NumpyInitializer(array)


def random_uniform(shape, minval: float = -0.05, maxval: float = 0.05, dtype=np.float32):
    array = np.random.uniform(size=shape, low=minval, high=maxval).astype(dtype)
    return dy.NumpyInitializer(array)


def truncated_normal(shape, mean: float = 0.0, stddev: float = 1.0, dtype=np.float32):
    array = np.random.normal(size=shape, loc=mean, scale=stddev).astype(dtype)
    np.clip(array, mean - 2 * stddev, mean + 2 * stddev, array)
    return dy.NumpyInitializer(array)


# TODO VarianceScaling

def orthogonal(shape, scale: float = 1.0, dtype=np.float32):
    array = initializers.generate_array(initializers.Orthogonal(scale=scale, dtype=dtype), shape=shape, xp=np)
    return dy.NumpyInitializer(array)


def identity(shape, scale: float = 1.0, dtype=np.float32):
    array = initializers.generate_array(initializers.Identity(scale, dtype=dtype), shape=shape, xp=np)
    return dy.NumpyInitializer(array)


def lecun_uniform(shape, scale: float = 1.0, dtype=np.float32):
    array = initializers.generate_array(initializers.LeCunUniform(scale=scale, dtype=dtype), shape=shape, xp=np)
    return dy.NumpyInitializer(array)


def lecun_normal(shape, scale: float = 1.0, dtype=np.float32):
    array = initializers.generate_array(initializers.LeCunNormal(scale=scale, dtype=dtype), shape=shape, xp=np)
    return dy.NumpyInitializer(array)


def glorot_uniform(shape, scale: float = 1.0, dtype=np.float32):
    array = initializers.generate_array(initializers.GlorotUniform(scale=scale, dtype=dtype), shape=shape, xp=np)
    return dy.NumpyInitializer(array)


xavier = glorot_uniform


def glorot_normal(shape, scale: float = 1.0, dtype=np.float32):
    array = initializers.generate_array(initializers.GlorotNormal(scale=scale, dtype=dtype), shape=shape, xp=np)
    return dy.NumpyInitializer(array)


def he_uniform(shape, scale: float = 1.0, dtype=np.float32):
    array = initializers.generate_array(initializers.HeUniform(scale=scale, dtype=dtype), shape=shape, xp=np)
    return dy.NumpyInitializer(array)


def he_normal(shape, scale: float = 1.0, dtype=np.float32):
    array = initializers.generate_array(initializers.HeNormal(scale=scale, dtype=dtype), shape=shape, xp=np)
    return dy.NumpyInitializer(array)


if __name__ == '__main__':
    from moire import Expression, ParameterCollection

    pc: dy.Model = ParameterCollection()

    w = pc.add_parameters((10, 10), he_uniform((10, 10)))
    print(w.as_array())
