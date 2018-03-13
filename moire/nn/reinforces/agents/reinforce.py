import warnings

import dynet as dy
import numpy as np

from moire import nn
from moire.nn.indexing import gumbel_argmax, argmax
from moire.nn.informatics import entropy
from .agent import Agent


class REINFORCE(Agent):
    def __init__(self, policy: nn.Module, optimizer: dy.Trainer,
                 beta: float = 1e-4,
                 batch_size: int = 32,
                 act_deterministically: bool = True,
                 average_entropy_decay: float = 0.999,
                 backward_separately: bool = False, indeterministic_argmax=gumbel_argmax):
        super(REINFORCE, self).__init__()

        self.policy = policy
        self.optimizer = optimizer
        self.beta = beta
        self.batch_size = batch_size
        self.act_deterministically = act_deterministically
        self.indeterministic_argmax = indeterministic_argmax
        self.backward_separately = backward_separately
        self.average_entropy_decay = average_entropy_decay

        # Statistics
        self.average_entropy = 0

        self.t = 0
        self.obs_sequences = [[]]
        self.action_sequences = [[]]
        self.reward_sequences = [[]]
        self.n_backward = 0

    def act_and_train(self, obs, reward):

        obs = dy.nobackprop(obs)
        action_dist = dy.softmax(self.policy(obs))
        action = self.indeterministic_argmax(action_dist)

        # Save values used to compute losses
        self.obs_sequences[-1].append(obs.npvalue())
        self.action_sequences[-1].append(action)
        self.reward_sequences[-1].append(reward)

        self.t += 1

        # Update stats
        self.average_entropy += (
                (1 - self.average_entropy_decay) *
                (float(entropy(action_dist).value()) - self.average_entropy))

        return action

    def act(self, obs):
        obs = dy.nobackprop(obs)
        action_dist = dy.softmax(self.policy(obs))
        if self.act_deterministically:
            return argmax(action_dist)
        return self.indeterministic_argmax(action_dist)

    def stop_episode_and_train(self, obs, reward, done=False):

        if not done:
            warnings.warn(
                'Since REINFORCE supports episodic environments only, '
                'calling stop_episode_and_train with done=False will throw '
                'away the last episode.')
            self.obs_sequences[-1] = []
            self.action_sequences[-1] = []
            self.reward_sequences[-1] = []
        else:
            self.reward_sequences[-1].append(reward)
            if self.backward_separately:
                self.accumulate_grad()
                if self.n_backward == self.batch_size:
                    self.update_with_accumulated_grad()
            else:
                if len(self.reward_sequences) == self.batch_size:
                    self.batch_update()
                else:
                    # Prepare for the next episode
                    self.reward_sequences.append([])
                    self.obs_sequences.append([])
                    self.action_sequences.append([])

    def accumulate_grad(self):
        if self.n_backward == 0:
            self.policy.zerograds()
        # Compute losses
        losses = []
        for r_seq, obs_seq, action_seq in zip(self.reward_sequences,
                                              self.obs_sequences,
                                              self.action_sequences):
            assert len(r_seq) - 1 == len(obs_seq) == len(action_seq)
            # Convert rewards into returns (=sum of future rewards)
            R_seq = np.cumsum(list(reversed(r_seq[1:])))[::-1]
            for R, obs, action in zip(R_seq, obs_seq, action_seq):
                action_dist = dy.softmax(self.policy(dy.inputVector(obs)))
                loss = -R * dy.log(dy.pick(action_dist, action)) - self.beta * entropy(action_dist)
                losses.append(loss)
        # When self.batch_size is future.types.newint.newint, dividing a
        # Variable with it will raise an error, so it is manually converted to
        # float here.
        dy.average(losses).backward()
        self.reward_sequences = [[]]
        self.obs_sequences = [[]]
        self.action_sequences = [[]]
        self.n_backward += 1

    def batch_update(self):
        assert len(self.reward_sequences) == self.batch_size
        assert len(self.obs_sequences) == self.batch_size
        assert len(self.action_sequences) == self.batch_size
        # Update the model
        self.policy.zerograds()
        self.accumulate_grad()
        self.optimizer.update()
        self.n_backward = 0

    def update_with_accumulated_grad(self):
        assert self.n_backward == self.batch_size
        self.optimizer.update()
        self.n_backward = 0

    def stop_episode(self):
        pass
