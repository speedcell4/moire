import dynet as dy

import moire
from moire import ParameterCollection
from moire import nn
from moire.nn.indexing import argmax
from moire.nn.reinforces.agents import DQN
from moire.nn.reinforces.replay_buffer import ReplayBuffer


class DoubleDQN(DQN):
    def _compute_target_values(self, exp_instance, gamma):
        state, action, reward, next_state, _, done = exp_instance

        next_q_out = self.q_function(next_state)
        target_next_q_out = self.target_q_function(next_state)
        max_index = argmax(next_q_out)
        next_q = dy.pick(target_next_q_out, max_index)

        return reward + gamma * (1.0 - done) * next_q


if __name__ == '__main__':
    moire.config.epsilon = 0.7

    pc = ParameterCollection()
    q_function = nn.MLP(pc, 2, 3, 5)
    target_q_function = nn.MLP(ParameterCollection(), 2, 3, 5)

    optimizer = dy.AdamTrainer(pc)

    dqn = DoubleDQN(q_function, target_q_function, ReplayBuffer(), optimizer, 0.8, 32, 50, 1, 1, 100)

    for _ in range(10000):
        for _ in range(12):
            dqn.act_and_train(dy.inputVector([1, 2, 3]), 0.8)
        dqn.stop_episode_and_train(dy.inputVector([1, 2, 3]), 0.7, True)

        moire.debug(f'average_q => {dqn.average_q:.03f}, average_l => {dqn.average_loss:.03f}',
                    file=moire.config.stdlog)
