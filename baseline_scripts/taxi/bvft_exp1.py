import numpy as np
from Q_learning import Q_learning

from environment import random_walk_2d, taxi
import os

start = 70

def train_Q_learning(env, num_trajectory, truncate_size, temperature=2.0):
    agent = Q_learning(n_state, n_action, 0.005, 0.99)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    state = env.reset()
    agent.Q = np.load(dir_path + '/taxi-q/q{}.npy'.format(start-1))
    cum_rewards = []
    for k in range(start, start + 30):
        print('Training for episode {}'.format(k))
        data_mem = []
        for i in range(50):
            for j in range(5000):
                action = agent.choose_action(state, temperature)
                next_state, reward = env.step(action)
                agent.update(state, action, next_state, reward)
                data_mem.append([state, action, reward, next_state])
                state = next_state
        pi = agent.get_pi(temperature)
        Q = agent.Q
        np.save(dir_path + '/taxi-q/q{}.npy'.format(k), Q)
        np.save(dir_path + '/taxi-d/d{}.npy'.format(k), np.array(data_mem, dtype=int))

        SAS, f, avr_reward = roll_out(n_state, env, pi, num_trajectory, truncate_size)
        cum_rewards.append(avr_reward)
        np.save(dir_path + '/taxi-q/rewards2.npy', np.array(cum_rewards))
        print('Episode {} reward = {}'.format(k, avr_reward))
        # heat_map(length, f, env, 'heatmap/pi{}.pdf'.format(k))


def roll_out(state_num, env, policy, num_trajectory, truncate_size):
    SASR = []
    total_reward = 0.0
    frequency = np.zeros(state_num)
    for i_trajectory in range(num_trajectory):
        state = env.reset()
        sasr = []
        for i_t in range(truncate_size):
            # env.render()
            p_action = policy[state, :]
            action = np.random.choice(p_action.shape[0], 1, p=p_action)[0]
            next_state, reward = env.step(action)

            sasr.append((state, action, next_state, reward))
            frequency[state] += 1
            total_reward += reward
            # print env.state_decoding(state)
            # a = input()

            state = next_state
        SASR.append(sasr)
    return SASR, frequency, total_reward / (num_trajectory * truncate_size)


if __name__ == '__main__':
    length = 5
    env = taxi(length)
    n_state = env.n_state
    n_action = env.n_action
    num_trajectory = 200
    truncate_size = 400
    gamma = 0.95

    train_Q_learning(env, num_trajectory, truncate_size)