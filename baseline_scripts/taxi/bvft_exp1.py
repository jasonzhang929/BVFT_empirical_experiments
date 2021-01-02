import numpy as np
from Q_learning import Q_learning
import numpy as np
from environment import taxi
import os


def train_Q_learning(env, alpha=0.005, temperature=2.0):
    agent = Q_learning(n_state, n_action, alpha, 0.99)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    state = env.reset()
    # agent.Q = np.load(dir_path + '/../../data/taxi/q{}.npy'.format(start-1))
    start = np.random.randint(1e6)
    for k in range(start, start + 40):
        print('Training for episode {}'.format(k))
        data_mem = []
        for i in range(50):
            for j in range(5000):
                action = agent.choose_action(state, temperature)
                next_state, reward = env.step(action)
                agent.update(state, action, next_state, reward)
                data_mem.append([state, action, reward, next_state])
                state = next_state

        Q = agent.Q
        if k % 4 == 0:
            np.save(dir_path + '/../../data/taxi/q{}.npy'.format(k), Q)


if __name__ == '__main__':
    length = 5
    env = taxi(length)
    n_state = env.n_state
    n_action = env.n_action
    num_trajectory = 200
    truncate_size = 400
    gamma = 0.95

    alpha = np.linspace(0.005, 0.02, 10)
    for i in range(10):
        train_Q_learning(env, alpha=alpha[i])