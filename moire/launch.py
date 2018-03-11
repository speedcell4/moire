import sys
import warnings

dynet_available = False


def _add_option(key, value):
    if value is not None:
        sys.argv.append(f'{key}')
        sys.argv.append(f'{value}')


def launch_dynet(device: str = 'CPU', memory: int = 1500, random_seed: int = None,
                 weight_decay: float = None, autobatch: int = None, profiling: int = None):
    _add_option(f'--dynet-devices', device)
    _add_option(f'--dynet-seed', random_seed)
    _add_option(f'--dynet-mem', memory)
    _add_option(f'--dynet-autobatch', autobatch)
    _add_option(f'--dynet-profiling', profiling)
    _add_option(f'--dynet-weight-decay', weight_decay)

    import random
    random.seed(random_seed)

    try:
        import numpy
        numpy.random.seed(random_seed)
    except ModuleNotFoundError:
        warnings.warn(r'numpy not found')

    import dynet

    if weight_decay is not None:
        print(f'[dynet] weight decay: {weight_decay:f}', file=sys.stderr)

    global dynet_available
    dynet_available = True
    return dynet


if __name__ == '__main__':
    launch_dynet(random_seed=2333, weight_decay=1e-6)
