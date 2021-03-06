from datetime import datetime
from pathlib import Path

import dynet as dy

from moire import nn
from moire.nn.reinforces.agents import Agent
from moire import Expression
from moire.utils import repeat


class Schedule(object):
    iteration: int

    def __init__(self, out_root: Path) -> None:
        super(Schedule, self).__init__()
        self.out_root = out_root

    @property
    def out_dir(self) -> Path:
        raise NotImplementedError

    def register_extension(self, extension: 'Extension') -> None:
        raise NotImplementedError


class Extension(object):
    def __call__(self, schedule: 'Schedule') -> None:
        moire.info(f'{self.__class__.__name__} is called')


class EpochalSchedule(Schedule):
    def __init__(self, module: nn.Module, iterator, optimizer: dy.Trainer, out_root: Path) -> None:
        super().__init__(out_root)
        self.module = module
        self.train_iterator = iterator
        self.optimizer = optimizer

        self.epoch: int = 0
        self.iteration: int = 0

        self.epoch_extensions = {
            True: [], False: []
        }
        self.iteration_extensions = {
            True: [], False: []
        }

    def register_extension(self, epoch: int = None, iteration: int = None, before: bool = False):
        assert bool(epoch is None) != bool(iteration is None), \
            'one and only one of epoch and iteration should be None'

        def wrapper(extension: Extension) -> None:
            if epoch is not None:
                def apply_extension(schedule: 'EpochalSchedule'):
                    if schedule.epoch % epoch == 0:
                        return extension.__call__(schedule)

                self.epoch_extensions[before].append(apply_extension)
            elif iteration is not None:
                def apply_extension(schedule: 'EpochalSchedule'):
                    if schedule.iteration % iteration == 0:
                        return extension.__call__(schedule)

                self.iteration_extensions[before].append(apply_extension)

        return wrapper

    def _train_enter_epoch(self) -> None:
        self.epoch += 1
        self.epoch_start_tm = datetime.now()

        for extension in self.epoch_extensions[True]:
            extension.__call__(self)

    def _train_finish_epoch(self) -> None:
        for extension in self.epoch_extensions[False]:
            extension.__call__(self)
        moire.info(
            f'[{moire.config.chapter} epoch-{self.epoch}]'
            f' elapsed: {datetime.now() - self.epoch_start_tm}')

    def _train_enter_iteration(self) -> None:
        self.iteration += 1

        for extension in self.iteration_extensions[True]:
            extension.__call__(self)

    def _train_finish_iteration(self) -> None:
        for extension in self.iteration_extensions[False]:
            extension.__call__(self)

        self.loss.backward()
        self.optimizer.update()
        del self.loss

    def train(self, nb_epoch: int, chapter: str = 'train', train: bool = True, debug: bool = False):
        with moire.using_config(chapter=chapter, train=train, debug=debug):
            for _ in range(nb_epoch):
                self._train_enter_epoch()
                for batch in self.train_iterator:
                    self._train_enter_iteration()
                    self.loss: Expression = self.module.__call__(batch)
                    self._train_finish_iteration()
                self._train_finish_epoch()

    def _eval_enter_epoch(self) -> None:
        raise NotImplementedError

    def _eval_finish_epoch(self) -> None:
        raise NotImplementedError

    def _eval_enter_iteration(self) -> None:
        raise NotImplementedError

    def _eval_finish_iteration(self) -> None:
        raise NotImplementedError

    def eval(self, nb_epoch: int) -> None:
        raise NotImplementedError


class EpisodicSchedule(Schedule):
    def __init__(self, agent: Agent, env, out_root: Path) -> None:
        super().__init__(out_root)
        self.agent = agent
        self.env = env

        self.step: int = 0
        self.episode: int = 0
        self.iteration: int = 0

        self.episode_extensions = {
            True: [], False: []
        }
        self.iteration_extensions = {
            True: [], False: []
        }

    def register_extension(self, episode: int = None, iteration: int = None, before: bool = False):
        assert bool(episode is None) != bool(iteration is None), \
            'one and only one of episode and iteration should be None'

        def wrapper(extension: Extension) -> None:
            if episode is not None:
                def apply_extension(schedule: 'EpisodicSchedule'):
                    if schedule.episode % episode == 0:
                        return extension.__call__(schedule)

                self.episode_extensions[before].append(apply_extension)
            elif iteration is not None:
                def apply_extension(schedule: 'EpisodicSchedule'):
                    if schedule.iteration % iteration == 0:
                        return extension.__call__(schedule)

                self.iteration_extensions[before].append(apply_extension)

        return wrapper

    def _train_enter_episode(self) -> None:
        self.episode += 1

        self.reward = 0
        self.done = False

        self.step = 0
        self.interval_start_tm = datetime.now()
        self.obs = self.env.reset()

        for extension in self.episode_extensions[True]:
            extension.__call__(self)

    def _train_finish_episode(self) -> None:
        for extension in self.episode_extensions[False]:
            extension.__call__(self)

    def _train_enter_iteration(self) -> None:
        self.iteration += 1

        for extension in self.iteration_extensions[True]:
            extension.__call__(self)

    def _train_finish_iteration(self) -> None:
        for extension in self.iteration_extensions[False]:
            extension.__call__(self)

    def train(self, nb_episodes: int = None, max_steps: int = None):
        for _ in repeat(None, times=nb_episodes):
            self._train_enter_episode()
            for _ in repeat(None, times=max_steps):
                if self.done:
                    break
                self._train_enter_iteration()
                self.action = self.agent.act_and_train(self.obs, self.reward)
                self.obs, self.reward, self.done, *self.info = self.env.step(self.action)
                self._train_finish_iteration()
                print(self.action)
                print(self.obs)
                print(self.reward)
                print(self.done)
            self.agent.stop_episode_and_train(self.obs, self.reward, self.done)
            self._train_finish_episode()

    def eval(self, nb_episodes: int):
        raise NotImplementedError


from .monitor import *
from .epsilon import *
from .history import *
