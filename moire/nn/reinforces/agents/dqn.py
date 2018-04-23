import dynet as dy
import numpy as np

import moire
from moire import Expression
from moire import ParameterCollection
from moire import nn
from moire.nn.indexing import argmax, epsilon_argmax
from moire.nn.reinforces.agents import Agent
from moire.nn.reinforces.replay_buffer import ReplayBuffer, ReplayUpdater, batch_experiences


class DQN(Agent):
    def __init__(self, q_function: nn.Module, target_q_function: nn.Module, replay_buffer: ReplayBuffer,
                 optimizer: dy.Trainer, gamma: float, batch_size: int, replay_start_size: int, n_times_update,
                 update_interval: int, target_update_interval: int,
                 average_q_decay: float = 0.999, average_loss_decay: float = 0.99) -> None:
        self.target_update_interval = target_update_interval
        self.optimizer = optimizer
        self.gamma = gamma
        self.q_function = q_function
        self.target_q_function = target_q_function

        self.replay_buffer = replay_buffer
        self.replay_updater = ReplayUpdater(
            replay_buffer=replay_buffer,
            update_func=self.update,
            batch_size=batch_size,
            n_times_update=n_times_update,
            replay_start_size=replay_start_size,
            update_interval=update_interval)

        self.t = 0
        self.last_obs = None
        self.last_action = None

        self.sync_target_network()

        self.average_q = 0
        self.average_q_decay = average_q_decay
        self.average_loss = 0
        self.average_loss_decay = average_loss_decay

    def sync_target_network(self):
        self.target_q_function.copy_from(self.q_function)
        moire.info(f'{self.__class__.__name__} synchronized at iteration :: {self.t}')

    def update(self, experiences):
        exp_batch = batch_experiences(experiences)
        loss = self._compute_loss(exp_batch, self.gamma)

        self.average_loss *= self.average_loss_decay
        self.average_loss += (1 - self.average_loss_decay) * float(loss.value())

        loss.backward()
        self.optimizer.update()

    def _compute_loss(self, exp_batch, gamma: float) -> Expression:
        losses = []
        for exp_instance in exp_batch:
            y, t = self._compute_y_and_t(exp_instance, gamma)
            losses.append(dy.huber_distance(y, t, 1.0))

        return dy.average(losses)

    def _compute_y_and_t(self, exp_instance, gamma):
        state, action, reward, next_state, _, done = exp_instance

        q = dy.pick(self.q_function(state), action)
        q_target = self._compute_target_values(exp_instance, gamma)

        return q, dy.nobackprop(q_target)

    def _compute_target_values(self, exp_instance, gamma):
        state, action, reward, next_state, _, done = exp_instance

        target_next_q_out = self.target_q_function(next_state)
        max_index = argmax(target_next_q_out)
        next_q = dy.pick(target_next_q_out, max_index)

        return reward + gamma * (1.0 - done) * next_q

    def act(self, obs):
        if isinstance(obs, Expression):
            obs = dy.nobackprop(obs)
        else:
            obs = dy.inputVector(obs)
        action_value = self.q_function(obs)
        action = argmax(action_value)
        q = float(dy.pick(action_value, action).value())

        self.average_q *= self.average_q_decay
        self.average_q += (1 - self.average_q_decay) * q

        return action

    def act_and_train(self, obs, reward):
        if moire.config.debug:
            moire.info(f'act_and_train :: {obs} => {reward}')

        if isinstance(obs, Expression):
            obs = dy.nobackprop(obs)
        else:
            obs = dy.inputVector(obs)
        action_value = self.q_function(obs)
        action = epsilon_argmax(action_value, moire.config.epsilon)
        q = float(dy.pick(action_value, action).value())

        self.average_q *= self.average_q_decay
        self.average_q += (1 - self.average_q_decay) * q

        self.t += 1
        if self.t % self.target_update_interval == 0:
            self.sync_target_network()

        if self.last_obs is not None:
            assert self.last_action is not None

            self.replay_buffer.append(
                state=self.last_obs,
                action=self.last_action,
                next_state=obs.npvalue(),
                next_action=action,
                reward=float(reward),
                is_state_terminal=0.0,
            )

        self.last_obs = obs.npvalue()
        self.last_action = action

        self.replay_updater.update_if_necessary(self.t)

        return action

    def stop_episode_and_train(self, obs, reward: float, done: bool):
        assert self.last_obs is not None
        assert self.last_action is not None

        if isinstance(obs, Expression):
            obs = obs.npvalue()
        else:
            obs = np.array(obs)

        if moire.config.debug:
            moire.info(f'stop_episode_and_train :: {obs} => {reward}, {done}')

        self.replay_buffer.append(
            state=self.last_obs,
            action=self.last_action,
            reward=float(reward),
            next_state=obs,
            next_action=self.last_action,
            is_state_terminal=float(done),
        )

        self.stop_episode()

    def stop_episode(self):
        self.last_obs = None
        self.last_action = None
        self.replay_buffer.stop_current_episode()


if __name__ == '__main__':
    moire.config.epsilon = 0.7

    pc = ParameterCollection()
    q_function = nn.MLP(pc, 2, 3, 5)
    target_q_function = nn.MLP(ParameterCollection(), 2, 3, 5)

    optimizer = dy.AdamTrainer(pc)

    dqn = DQN(q_function, target_q_function, ReplayBuffer(), optimizer, 0.8, 32, 50, 1, 1, 100)

    for _ in range(10000):
        for _ in range(12):
            dqn.act_and_train(dy.inputVector([1, 2, 3]), 0.8)
        dqn.stop_episode_and_train(dy.inputVector([1, 2, 3]), 0.7, True)

        moire.debug(f'average_q => {dqn.average_q:.03f}, average_l => {dqn.average_loss:.03f}',
                    file=moire.config.stdlog)
