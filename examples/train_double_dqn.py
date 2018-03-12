import aku
import gym

import launch_moire

app = aku.App(__file__)


@app.register
def train(device: str = 'CPU', gamma: float = 0.95, batch_size: int = 32, replay_start_size: int = 500,
          target_update_interval: int = 100, start_epsilon: float = 0.3, n_episodes: int = 200,
          max_episode_len: int = 2000,
          capacity: int = 10 ** 6, num_layers: int = 3,
          hidden_size: int = 50, update_interval: int = 1):
    launch_moire.launch_moire(device)

    import moire
    import dynet as dy
    from moire import ParameterCollection
    from moire import nn
    from moire.nn.reinforces import DoubleDQN, ReplayBuffer

    moire.config.epsilon = start_epsilon

    pc = ParameterCollection()
    q_function = nn.MLP(pc, num_layers, 4, 2, hidden_size)

    optimizer = dy.AdamTrainer(pc, eps=1e-3)

    target_q_function = nn.MLP(ParameterCollection(), num_layers, 4, 2, hidden_size)
    replay_buffer = ReplayBuffer(capacity)

    agent = DoubleDQN(q_function, target_q_function, replay_buffer, optimizer, gamma, batch_size,
                      replay_start_size, 1, update_interval, target_update_interval)

    env = gym.make('CartPole-v0')
    for i in range(1, n_episodes + 1):
        obs = env.reset()
        reward = 0
        done = False
        R = 0  # return (sum of rewards)
        t = 0  # time step

        while not done and t < max_episode_len:
            dy.renew_cg()
            env.render()
            action = agent.act_and_train(dy.inputVector(obs), reward)
            obs, reward, done, _ = env.step(action)

            R += reward
            t += 1
        if i % 10 == 0:
            print('episode:', i,
                  'R:', R,
                  f'average_q: {agent.average_q:.03f}',
                  f'average_loss: {agent.average_loss:.03f}')
        agent.stop_episode_and_train(dy.inputVector(obs), reward, done)
    print('Finished.')


if __name__ == '__main__':
    app.run()
