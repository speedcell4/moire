import moire
from moire.nn.scheduling import EpisodicSchedule, Extension

from moire.utils import clip

__all__ = [
    'ConstEpsilon',
    'LinearDecayEpsilon',
]


class ConstEpsilon(Extension):
    def __init__(self, epsilon: float):
        super(ConstEpsilon, self).__init__()
        self.epsilon = epsilon

    def __call__(self, schedule: 'EpisodicSchedule') -> None:
        super().__call__(schedule)
        moire.config.epsilon = self.epsilon
        moire.info(f'epsilon := {moire.config.epsilon}')


class LinearDecayEpsilon(Extension):
    def __init__(self, epsilon: float, decay: float, min_epsilon: float = None, max_epsilon: float = None) -> None:
        super(LinearDecayEpsilon, self).__init__()
        self.decay = decay
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon

    def __call__(self, schedule: 'EpisodicSchedule') -> None:
        super().__call__(schedule)
        if 'epsilon' not in moire.config.epsilon:
            moire.config.epsilon = self.epsilon
        else:
            moire.config.epsilon -= self.decay
        moire.config.epsilon = clip(moire.config.epsilon, min_value=self.min_epsilon, max_value=self.max_epsilon)
        moire.info(f'epsilon := {moire.config.epsilon}')
