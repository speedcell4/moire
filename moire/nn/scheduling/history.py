import moire
from moire.nn.scheduling import Schedule, Extension


class LogHistory(Extension):
    def __init__(self, name: str):
        super(LogHistory, self).__init__()
        self.name = name

    def __call__(self, schedule: 'Schedule') -> None:
        super(LogHistory, self).__call__(schedule)
        if self.name in moire.config:
            value = moire.summary_scalar(self.name)
            moire.notice(f'{self.name} => {value}')
