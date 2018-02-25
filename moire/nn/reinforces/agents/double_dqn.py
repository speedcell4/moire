import dynet as dy
from dynet import Trainer

from moire import nn, ParameterCollection, Expression
from moire.nn.indexing import EpsilonArgmax
from moire.nn.reinforces.agents import AgentMixin
from moire.nn.reinforces.replay_buffer import ReplayBuffer
from moire.nn.utils import to_numpy


class DoubleDQN(AgentMixin, nn.Module):
    def __init__(self, pc: ParameterCollection, optimizer: Trainer,
                 online_q_function, target_q_function,
                 capacity: int, nb_examples: int, interval: int, epsilon: float, gamma: float,
                 loss_fn=dy.huber_distance):
        super(DoubleDQN, self).__init__(pc)

        self.loss_fn = loss_fn
        self.interval = interval
        self.nb_examples = nb_examples

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_argmax = EpsilonArgmax(epsilon)

        self.optimizer = optimizer
        self.online_q_function = online_q_function
        self.target_q_function = target_q_function

        self.experiences = []
        self.replay_buffer = ReplayBuffer(capacity)

        self.last_obs = None
        self.last_action = None

    def clear_replay_buffer(self):
        self.replay_buffer.clear()

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} :: {self.epsilon}, {self.gamma}>'

    def __call__(self, x: Expression):
        y = self.online_q_function(x)
        return self.epsilon_argmax.__call__(y)

    def act(self, obs: Expression, reward: float, terminal: bool) -> int:
        action = self.__call__(obs)

        if not self.training:
            return action

        np_obs = to_numpy(obs)

        if self.last_obs is not None:
            self.replay_buffer.append(
                self.last_obs,
                self.last_action,
                np_obs,
                action,
                float(reward),
                float(terminal),
            )
            self.experiences.extend(
                self.replay_buffer.sample(self.nb_examples))
            if self.experiences.__len__() >= self.interval:
                self.end_episode_and_train()

        if not terminal:
            self.last_obs = np_obs
            self.last_action = action
        else:
            self.last_obs = None
            self.last_action = None

        return action

    def end_episode_and_train(self):
        if self.experiences.__len__() > 0:
            experience_losses = []
            for obs, action, next_obs, next_action, reward, terminal in self.experiences:
                obs = dy.inputVector(obs)
                next_obs = dy.inputVector(next_obs)

                y = dy.pick(self.target_q_function(obs), action)
                r = reward + terminal * self.gamma * dy.pick(self.target_q_function(next_obs), next_action)

                experience_losses.append(self.loss_fn(y, r))
            q_loss = dy.average(experience_losses)
            q_loss.backward()
            self.optimizer.update()
        self.experiences = []
