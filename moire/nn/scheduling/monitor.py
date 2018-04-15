from moire import report_scalar

from moire.nn.scheduling import Extension, Schedule

__all__ = [
    'MonitorSchedule', 'RewardMonitor',
]


class MonitorSchedule(Extension):
    def __init__(self, name: str, delete: bool = False):
        super(MonitorSchedule, self).__init__()
        self.name = name
        self.delete = delete

    def __call__(self, schedule: 'Schedule') -> None:
        super().__call__(schedule)
        value = getattr(self, self.name)
        report_scalar(self.name, value, schedule.iteration)


class RewardMonitor(MonitorSchedule):
    def __init__(self, delete: bool = False):
        super().__init__('reward', delete)
