from IPython.display import clear_output
import numpy as np
import random, os, time
import gym
from random_environment import TaxiRandom
from evaluate_q import roll_out

rand = 0.3
dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir_path = dir_path + F'/../../data/taxi-random-{rand}/'


class Q_learning(object):
    def __init__(self, n_state, n_action, alpha, gamma, epsilon_target, epsilon_steps):
        self.n_state = n_state
        self.n_action = n_action
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_target = epsilon_target
        self.epsilon_decay = (1.0 - self.epsilon_target) / epsilon_steps
        self.Q = (np.random.rand(n_state, n_action) - 0.5) * 300.0

    def update(self, s, a, sNext, r):
        self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + self.alpha * (r + self.gamma * np.max(self.Q[sNext]))
        if self.epsilon > self.epsilon_target:
            self.epsilon -= self.epsilon_decay

    def choose_action(self, s):
        if np.random.rand() > self.epsilon:
            return np.argmax(self.Q[s])
        else:
            return np.random.randint(self.n_action)



def train_Q_learning(env, alpha=0.005, epsilon_target=0.2, epsilon_steps=50000):
    agent = Q_learning(n_state, n_action, alpha, gamma, epsilon_target, epsilon_steps)

    start = np.random.randint(1e6)
    for k in range(start, start + 120):
        print('Training for episode {}'.format(k))
        for i in range(100):
            state = env.reset()
            for j in range(5000):
                action = agent.choose_action(state)
                next_state, reward = env.step(action)
                agent.update(state, action, next_state, reward)
                state = next_state

        Q = agent.Q
        if k % 5 == 0:
            value = roll_out(env, Q, gamma, 300, 500)
            print("value = {}, saved q{}.npy".format(value, k))
            np.save(data_dir_path + 'q{}.npy'.format(k), Q)


if __name__ == '__main__':
    length = 5
    env = TaxiRandom(length, rand)
    n_state = env.n_state
    n_action = env.n_action
    num_trajectory = 200
    truncate_size = 400
    gamma = 0.99

    alpha = np.linspace(0.005, 0.05, 10)
    for i in range(10):
        train_Q_learning(env, alpha=alpha[i])