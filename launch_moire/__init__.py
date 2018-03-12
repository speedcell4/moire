import sys
import warnings

__all__ = [
    'launch_moire',
]


def _option_value(key, value) -> None:
    if value is not None:
        sys.argv.append(f'--{key}')
        sys.argv.append(f'{value}')


def launch_moire(devices: str = 'CPU', memory: int = 1500, random_seed: int = 333,
                 weight_decay: float = 1e-6, autobatch: int = 0, profiling: int = 0):
    _option_value(f'dynet-mem', memory)
    _option_value(f'dynet-devices', devices)
    _option_value(f'dynet-seed', random_seed)
    _option_value(f'dynet-weight-decay', weight_decay)
    _option_value(f'dynet-autobatch', autobatch)
    _option_value(f'dynet-profiling', profiling)

    import moire
    moire.config.device = devices

    import random
    random.seed(random_seed)

    try:
        import numpy
        numpy.random.seed(random_seed)
    except ModuleNotFoundError:
        warnings.warn(r'numpy not found')

    if weight_decay is not None:
        print(f'[dynet] weight decay: {weight_decay:f}', file=sys.stderr)

    import dynet
    return dynet


if __name__ == '__main__':
    launch_moire(random_seed=2333, weight_decay=1e-6)
