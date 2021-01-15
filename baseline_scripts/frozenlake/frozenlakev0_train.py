from IPython.display import clear_output
import numpy as np
import random, os, time
import gym
from frozenlakev0_eval import roll_out


class Q_learning(object):
    def __init__(self, n_state, n_action, alpha, gamma, epsilon_target, epsilon_steps):
        self.n_state = n_state
        self.n_action = n_action
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_target = epsilon_target
        self.epsilon_decay = (1.0 - self.epsilon_target) / epsilon_steps
        self.Q = (np.random.rand(n_state, n_action)-0.5)*1

    def update(self, s, a, sNext, r, done):
        if done:
            self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + self.alpha * r
        else:
            self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + self.alpha * (r + self.gamma * np.max(self.Q[sNext]))
        if self.epsilon > self.epsilon_target:
            self.epsilon -= self.epsilon_decay

    def choose_action(self, s):
        if np.random.rand() > self.epsilon:
            return np.argmax(self.Q[s])
        else:
            return np.random.randint(self.n_action)



def train_Q_learning(env, alpha=0.005, max_episodes=int(1e6), max_steps=500, epsilon_target=0.1, epsilon_steps=500000):
    agent = Q_learning(n_state, n_action, alpha, gamma, epsilon_target, epsilon_steps)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    start = np.random.randint(1e6)

    for episode in range(max_episodes):
        state = env.reset()
        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, next_state, reward, done)
            if done:
                break
            state = next_state

        if (episode + 1) % 30000 == 0:
            Q = agent.Q
            value = roll_out(env, Q, gamma, 10000, 100)
            print(F"Episode {episode}, value = {value}")
            np.save(dir_path + '/../../data/frozenlake/q{}_{}_VALUE_{:.7}.npy'.format(start, episode+1, value), Q)


if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    n_state = env.observation_space.n
    n_action = env.action_space.n
    gamma = 0.95

    n = 5
    alpha = np.linspace(0.005, 0.02, n)
    for i in range(n):
        print(F"train with alpha {alpha[i]}")
        train_Q_learning(env, alpha=alpha[i], max_episodes=300000)
