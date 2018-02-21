from collections import OrderedDict

from dynet import Model

from moire import ParameterCollection, Parameters, LookupParameters


class Function(object):
    def __init__(self):
        self.training = False

    def __repr__(self):
        raise NotImplementedError

    def copy_from(self, other: 'Function'):
        pass

    def __call__(self, *inputs):
        raise NotImplementedError


# TODO Builders
# TODO save / load, pickle, copy, deepcopy
class Module(object):
    def __init__(self, pc: ParameterCollection):
        self.pc: Model = pc.add_subcollection()

        self._training = False

        self._modules = OrderedDict()
        self._functions = OrderedDict()
        self._parameters = OrderedDict()
        self._lookup_parameters = OrderedDict()

    def __repr__(self):
        raise NotImplementedError

    @property
    def training(self) -> bool:
        return self._training

    @training.setter
    def training(self, value: bool) -> None:
        if self._training != value:
            self._training = value
            for child in self.children:
                child.training = value

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
