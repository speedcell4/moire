import aku
import gym

import launch_moire

app = aku.App(__file__)


@app.register
def train(device: str = 'CPU', beta: float = 1e-4, average_entropy_decay: float = 0.999,
          backward_separately: bool = True, batch_size: int = 32, n_episodes: int = 2000,
          max_episode_len: int = 20000, num_layers: int = 3,
          hidden_size: int = 100):
    launch_moire.launch_moire(device)

    import dynet as dy
    from moire import ParameterCollection
    from moire import nn
    from moire.nn.thresholds import relu
    from moire.nn.reinforces.agents.reinforce import REINFORCE

    pc = ParameterCollection()
    policy = nn.MLP(pc, num_layers=num_layers, in_feature=4, out_feature=2, hidden_feature=hidden_size, nonlinear=relu)

    optimizer = dy.AdamTrainer(pc)

    agent = REINFORCE(
        policy=policy, optimizer=optimizer, beta=beta, average_entropy_decay=average_entropy_decay,
        backward_separately=backward_separately, batch_size=batch_size,
    )

    env = gym.make('CartPole-v0')
    for i in range(1, n_episodes + 1):
        obs = env.reset()
        reward = 0
        done = False
        R = 0  # return (sum of rewards)
        t = 0  # time step

        while not done and t < max_episode_len:
            dy.renew_cg()
            action = agent.act_and_train(dy.inputVector(obs), reward)
            obs, reward, done, _ = env.step(action)

            R += reward
            t += 1
        if i % 10 == 0:
            print('episode:', i,
                  'R:', R,
                  f'average_entropy: {agent.average_entropy:.03f}')
        agent.stop_episode_and_train(dy.inputVector(obs), reward, done)
    print('Finished.')


if __name__ == '__main__':
    app.run()
