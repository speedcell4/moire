import itertools
from collections import OrderedDict
from pathlib import Path
from typing import List
import warnings

import dynet as dy
import numpy as np

import moire
from moire import LookupParameters, ParameterCollection, Parameters


class Function(object):
    def __init__(self):
        super(Function, self).__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'

    @property
    def gain(self) -> float:
        raise AttributeError

    def copy_from(self, other: 'Function'):
        pass

    def __call__(self, *inputs):
        raise NotImplementedError


# TODO Builders
# TODO save / load, pickle, copy, deepcopy
class Module(object):
    def __init__(self, pc: ParameterCollection, sub_module: bool = True) -> None:
        super(Module, self).__init__()

        if sub_module:
            self.pc: dy.Model = pc.add_subcollection()
        else:
            self.pc: dy.Model = pc

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
            target.set_value(source.as_array())
        for target, source in zip(self.lookup_parameters, other.lookup_parameters):
            target.set_value(source.as_array())

    def soft_copy_from(self, other: 'Module', tau: float) -> None:
        for target, source in zip(self.functions, other.functions):
            target.copy_from(source)
        for target, source in zip(self.parameters, other.parameters):
            target.set_value(target.as_array() * (1 - tau) + source.as_array() * tau)
        for target, source in zip(self.lookup_parameters, other.lookup_parameters):
            target.set_value(target.as_array() * (1 - tau) + source.as_array() * tau)

    def scale_gradient(self, scale: float):
        for p in self.parameters:
            p.scale_gradient(scale)
        for p in self.lookup_parameters:
            p.scale_gradient(scale)

    def zerograds(self) -> None:
        self.scale_gradient(0.0)

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


class ModuleList(Module):
    def __init__(self, pc: ParameterCollection, modules: List[Module] = None) -> None:
        super().__init__(pc, sub_module=False)
        warnings.warn(f'{self.__class__.__name__} has not been tested', FutureWarning)
        if modules is not None:
            self.extend(modules)

    def append(self, module: Module):
        ix = self._modules.__len__()
        return setattr(self, f'layer{ix}', module)

    def extend(self, modules: List[Module]):
        for module in modules:
            self.append(module)

    def __getitem__(self, ix: int) -> Module:
        return self._modules[f'layer{ix}']

    def __call__(self, *inputs):
        raise NotImplementedError(f'{self.__class__.__name__} is not callable')
