class Agent(object):
    def sync_target_network(self):
        raise NotImplementedError

    def act(self, obs):
        raise NotImplementedError

    def act_and_train(self, obs, reward: float):
        raise NotImplementedError

    def stop_episode(self):
        raise NotImplementedError

    def stop_episode_and_train(self, obs, reward: float, done: bool):
        raise NotImplementedError
