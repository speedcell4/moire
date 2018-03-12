import dynet as dy

from moire.nn.indexing import argmax
from moire.nn.reinforces.agents import DQN


class DoubleDQN(DQN):
    def _compute_target_values(self, exp_instance, gamma):
        state, action, reward, next_state, _, done = exp_instance

        next_q_out = self.q_function(next_state)
        target_next_q_out = self.target_q_function(next_state)
        max_index = argmax(next_q_out)
        next_q = dy.pick(target_next_q_out, max_index)

        return reward + gamma * (1.0 - done) * next_q
