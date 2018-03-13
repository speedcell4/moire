import contextlib
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
    stdlog=sys.stdout,
    stdout=sys.stdout,
    stderr=sys.stderr,

    chapter='train',
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


if __name__ == '__main__':
    print(config.train)
    with using_config(train=False):
        print(config.train)
    print(config.train)
