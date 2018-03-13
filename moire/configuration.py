import contextlib
import copy
import sys
from pathlib import Path
from typing import Optional


class Configuration(object):
    def __init__(self, **kwargs):
        super(Configuration, self).__init__()
        for name, value in kwargs.items():
            setattr(self, name, value)

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}>'


config = Configuration(
    train=True,
    debug=False,

    device='CPU',
    stdin=sys.stdin,
    stdlog=sys.stdout,
    stdout=sys.stdout,
    stderr=sys.stderr,

    chapter='train',
)


@contextlib.contextmanager
def using_config(**kwargs):
    global config
    old_config = copy.copy(copy)

    for name, value in kwargs.items():
        setattr(config, name, value)

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
