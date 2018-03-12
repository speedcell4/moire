import dynet as dy

__all__ = [
    'batch_experiences',
    'ReplayUpdater',
]


def batch_experiences(experiences):
    exp_batch = []
    for instance in experiences:
        state = instance['state']
        action = instance['action']
        next_state = instance['next_state']
        next_action = instance['next_action']
        reward = instance['reward']
        is_state_terminal = instance['is_state_terminal']

        exp_batch.append((
            dy.inputVector(state), int(action),
            float(reward),
            dy.inputVector(next_state), int(next_action),
            float(is_state_terminal),
        ))

    return exp_batch


class ReplayUpdater(object):
    def __init__(self, replay_buffer, update_func, batch_size,
                 n_times_update, replay_start_size, update_interval):

        assert batch_size <= replay_start_size

        self.replay_buffer = replay_buffer
        self.update_func = update_func
        self.batch_size = batch_size
        self.n_times_update = n_times_update
        self.replay_start_size = replay_start_size
        self.update_interval = update_interval

    def update_if_necessary(self, iteration):
        if len(self.replay_buffer) < self.replay_start_size:
            return

        if iteration % self.update_interval != 0:
            return

        for _ in range(self.n_times_update):
            transitions = self.replay_buffer.sample(self.batch_size)
            self.update_func(transitions)
