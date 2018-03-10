import contextlib
import copy
import sys


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
)


@contextlib.contextmanager
def using_config(**kwargs):
    global config
    old_config = copy.deepcopy(copy)

    for name, value in kwargs.items():
        setattr(config, name, value)

    try:
        yield
    finally:
        config = old_config
