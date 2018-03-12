import aku
import gym

import launch_moire

app = aku.App(__file__)


@app.register
def train(device: str = 'CPU', gamma: float = 0.95, batch_size: int = 32, replay_start_size: int = 500,
          target_update_interval: int = 100, start_epsilon: float = 0.3, epsilon_decay_interval: int = 50000,
          epsilon_decay: float = 0.03, nb_iterations: int = 200000000, capacity: int = 10 ** 6, num_layers: int = 2,
          hidden_size: int = 200):
    launch_moire.launch_moire(device)

    import moire
    import dynet as dy
    from moire import ParameterCollection
    from moire import nn
    from moire.nn.reinforces import DoubleDQN, ReplayBuffer

    moire.config.epsilon = start_epsilon

    pc = ParameterCollection()
    q_function = nn.MLP(pc, num_layers, 4, 2, hidden_size)

    optimizer = dy.AdamTrainer(pc)

    target_q_function = nn.MLP(ParameterCollection(), num_layers, 4, 2, hidden_size)
    replay_buffer = ReplayBuffer(capacity)

    agent = DoubleDQN(q_function, target_q_function, replay_buffer, optimizer, gamma, batch_size,
                      replay_start_size, 1, 5, target_update_interval)

    env = gym.make('CartPole-v0')
    for i_episode in range(1, nb_iterations):
        obs, reward, done = env.reset(), 0.0, False

        while not done:
            action = agent.act_and_train(dy.inputVector(obs), reward)
            obs, reward, done, _ = env.step(action)

        agent.stop_episode_and_train(dy.inputVector(obs), reward, done)

        if i_episode % epsilon_decay_interval == 0:
            moire.config.epsilon -= epsilon_decay
            print(f'epsilon => {moire.config.epsilon:.03f}', file=moire.config.stdlog)

        print(f'{i_episode:010d} :: average_q => {agent.average_q:.03f}, average_l => {agent.average_loss:.03f}', file=moire.config.stdlog)


if __name__ == '__main__':
    app.run()
