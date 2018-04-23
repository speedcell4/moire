from typing import Callable, Dict, Union

import numpy as np
import dynet as dy

Expression = dy.Expression
Activation = Callable[[Expression], Expression]

Parameters = dy.Parameters
LookupParameters = dy.LookupParameters
ParameterCollection = dy.ParameterCollection

from .array import *

import logbook
import contextlib
from collections import OrderedDict
import sys
from pathlib import Path
from typing import Optional


class Configuration(dict):
    def __init__(self, **kwargs):
        super(Configuration, self).__init__(**kwargs)

    def __getattr__(self, key: str):
        return super(Configuration, self).__getitem__(key)

    def __delattr__(self, key: str):
        return super(Configuration, self).__delitem__(key)

    def __setattr__(self, key: str, value):
        return super(Configuration, self).__setitem__(key, value)


config = Configuration(
    train=True,
    debug=False,

    device='CPU',
    stdin=sys.stdin,
    stdout=sys.stdout,
    stderr=sys.stderr,

    chapter='train',
    observation=OrderedDict(),
)


@contextlib.contextmanager
def using_config(**kwargs):
    global config
    old_config = Configuration(**config)
    config = Configuration(**{**config, **kwargs})

    try:
        yield
    finally:
        config = old_config


@contextlib.contextmanager
def redirect_stream(name: str, path: Optional[Path], mode: str = 'r', encoding: str = 'utf-8'):
    if path is None:
        with using_config(**{name: path}):
            try:
                yield
            finally:
                pass
    else:
        with path.expanduser().absolute().open(mode, encoding=encoding) as fp:
            with using_config(**{name: fp}):
                try:
                    yield
                finally:
                    pass


def report_scalar(name: str, value: Union[int, float], iteraiton: int = None) -> None:
    global config
    config.observation.setdefault(name, []).append((value, iteraiton))


def summary_scalar(name: str) -> float:
    global config
    values, _ = zip(*config.observation.get(name, [0.0]))
    mean = np.array(values, dtype=np.float32).mean()
    del config.observation[name]
    return mean


def summary_all_scalars() -> Dict[str, float]:
    global config
    return {
        name: summary_scalar(name)
        for name in config.observation.keys()
    }


logger = logbook.Logger(__name__, level=logbook.NOTICE)
logbook.StreamHandler(stream=sys.stdout, level=logbook.NOTICE).push_application()

CRITICAL = logbook.CRITICAL
ERROR = logbook.ERROR
WARNING = logbook.WARNING
NOTICE = logbook.NOTICE
INFO = logbook.INFO
DEBUG = logbook.DEBUG
TRACE = logbook.TRACE

critical = logger.critical
error = logger.error
warning = logger.warning
notice = logger.notice
info = logger.info
debug = logger.debug
trace = logger.trace
