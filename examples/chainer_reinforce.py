import aku
import gym

from chainer import optimizers

from chainerrl.agents import REINFORCE
from chainerrl.policies import FCSoftmaxPolicy

app = aku.App(__file__)


@app.register
def train(device: str = 'CPU', beta: float = 1e-4, average_entropy_decay: float = 0.999,
          backward_separately: bool = False, batch_size: int = 32, n_episodes: int = 2000,
          max_episode_len: int = 20000, num_layers: int = 3,
          hidden_size: int = 100):
    policy = FCSoftmaxPolicy(
        n_input_channels=4, n_hidden_layers=num_layers, n_hidden_channels=hidden_size,
        n_actions=2)
    optimizer = optimizers.Adam()
    optimizer.setup(policy)

    agent = REINFORCE(
        model=policy, optimizer=optimizer,
        beta=beta, batchsize=batch_size,
        average_entropy_decay=average_entropy_decay,
        backward_separately=backward_separately,
    )

    env = gym.make('CartPole-v0')
    for i in range(1, n_episodes + 1):
        obs = env.reset()
        reward = 0
        done = False
        R = 0  # return (sum of rewards)
        t = 0  # time step

        while not done and t < max_episode_len:
            action = agent.act_and_train(obs.astype('f'), reward)
            obs, reward, done, _ = env.step(action)

            R += reward
            t += 1
        if i % 10 == 0:
            print('episode:', i,
                  'R:', R,
                  f'average_entropy: {agent.average_entropy:.03f}')
        agent.stop_episode_and_train(obs.astype('f'), reward, done)
    print('Finished.')


if __name__ == '__main__':
    app.run()
