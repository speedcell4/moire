import itertools
from collections import OrderedDict
from pathlib import Path

import dynet as dy
import numpy as np

import moire
from moire import ParameterCollection, Parameters, LookupParameters


class Function(object):
    def __init__(self):
        super(Function, self).__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'

    def copy_from(self, other: 'Function'):
        pass

    def __call__(self, *inputs):
        raise NotImplementedError


# TODO Builders
# TODO save / load, pickle, copy, deepcopy
class Module(object):
    def __init__(self, pc: ParameterCollection):
        super(Module, self).__init__()

        self.pc: dy.Model = pc.add_subcollection()

        self._modules = OrderedDict()
        self._functions = OrderedDict()
        self._parameters = OrderedDict()
        self._lookup_parameters = OrderedDict()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'

    def add_param(self, shape, initializer):
        return self.pc.add_parameters(shape, initializer(shape), device=moire.config.device)

    def add_lookup(self, shape, initializer):
        return self.pc.add_lookup_parameters(shape, initializer(shape), device=moire.config.device)

    @property
    def children(self):
        yield from self.modules
        yield from self.functions

    @property
    def modules(self):
        yield from self._modules.values()

    @property
    def functions(self):
        yield from self._functions.values()

    @property
    def parameters(self):
        yield from self._parameters.values()
        for module in self.modules:
            yield from module.parameters

    @property
    def lookup_parameters(self):
        yield from self._lookup_parameters.values()
        for module in self.modules:
            yield from module.lookup_parameters

    def copy_from(self, other: 'Module') -> None:
        for target, source in zip(self.functions, other.functions):
            target.copy_from(source)
        for target, source in zip(self.parameters, other.parameters):
            target.set_value(source.value())
        for target, source in zip(self.lookup_parameters, other.lookup_parameters):
            target.set_value(source.value())

    def __setattr__(self, key, value):
        if isinstance(value, Module):
            self._modules[key] = value
        elif isinstance(value, Function):
            self._functions[key] = value
        elif isinstance(value, Parameters):
            self._parameters[key] = value
        elif isinstance(value, LookupParameters):
            self._lookup_parameters[key] = value
        return super(Module, self).__setattr__(key, value)

    def __call__(self, *inputs):
        raise NotImplementedError

    def save(self, path: Path) -> None:
        return np.save(path.expanduser().absolute().__str__(),
                       itertools.chain(self.parameters, self.lookup_parameters))

    def load(self, path: Path) -> None:
        values = np.load(path.expanduser().absolute().__str__())
        for parameter in self.parameters:
            parameter.set_value(next(values))
        for lookup_parameter in self.lookup_parameters:
            lookup_parameter.init_from_array(next(values))
