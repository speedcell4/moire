from collections import deque

import numpy as np

__all__ = [
    'ReplayBuffer',
]


class ReplayBuffer(object):
    def __init__(self, capacity: int = None):
        self.memory = deque(maxlen=capacity)

    def __len__(self):
        return self.memory.__len__()

    def clear(self):
        self.memory.clear()

    def append(self, *items):
        self.memory.append(items)

    def sample(self, n: int):
        ixs = np.random.sample(size=(n,)) * self.__len__()
        return [self.memory[ix] for ix in ixs.astype(np.int32).tolist()]


if __name__ == '__main__':
    replay_buffer = ReplayBuffer(capacity=2)
    replay_buffer.append(2)
    print(replay_buffer.sample(3))
    replay_buffer.append(1)
    print(replay_buffer.sample(3))
    replay_buffer.append(4)
    print(replay_buffer.sample(3))
    replay_buffer.append(3)
    print(replay_buffer.sample(3))
