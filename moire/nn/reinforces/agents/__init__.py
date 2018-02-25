from moire import Expression


class AgentMixin(object):
    def __init__(self, *args, **kwargs):
        super(AgentMixin, self).__init__(*args, **kwargs)

    def act(self, obs: Expression, reward: float, terminal: float):
        raise NotImplementedError

    def end_episode_and_train(self) -> Expression:
        raise NotImplementedError
