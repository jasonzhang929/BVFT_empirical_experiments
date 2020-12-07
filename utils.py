import matplotlib.pyplot as plt
import numpy as np
import os
from BVFT import BVFT


def plot_bars():
    gamma = 0.99
    rmax = 20.0
    rmin = -1.0

    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = []
    for i in range(25, 27):
        data.append(np.load(dir_path + '/taxi-d/d{}.npy'.format(i)))
    data = np.concatenate(data, axis=0)
    qs = []
    for i in range(70, 100):
        qs.append(np.load(dir_path + '/taxi-q/q{}.npy'.format(i)))
    avg_reward = (np.load(dir_path + '/taxi-q/rewards1.npy'))

    b = BVFT_discrete(data[:20], gamma, rmax, rmin, resolution=1e-3)
    ranks = b.run_BVFT(qs)[1]
    y_val = [avg_reward[i] for i in ranks]
    plt.bar([i for i in range(30)], y_val)
    plt.ylabel("rollout estimates")
    plt.xlabel("rank")
    plt.show()


def plot_bars1():
    gamma = 0.99
    rmax = 20.0
    rmin = -1.0
    k = 15

    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = []
    for i in range(80, 82):
        data.append(np.load(dir_path + '/taxi-d/d{}.npy'.format(i)))
    data = np.concatenate(data, axis=0)
    # models = [i for i in range(30)] + [i for i in range(40, 100)]
    models = [i for i in range(40, 70)]
    ids = random.sample(models, k)

    qs = []
    avg_rewards = []

    r = np.load(dir_path + '/taxi-q/rewards.npy')
    for id in ids:
        qs.append(np.load(dir_path + '/taxi-q/q{}.npy'.format(id)))
        avg_rewards.append(r[id])
    np.random.shuffle(data)
    b = BVFT_discrete(data[:5000], gamma, rmax, rmin, resolution=1e-3)
    ranks = b.run_BVFT(qs)[1]
    y_val = [avg_rewards[i]+1 for i in ranks]
    plt.bar([i for i in range(k)], y_val)
    plt.ylabel("rollout estimates")
    plt.xlabel("rank")
    plt.show()

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



def rank_correlation():
    pass


def group_roll_outs():
    length = 5
    env = taxi(length)
    n_state = env.n_state
    n_action = env.n_action
    num_trajectory = 200
    truncate_size = 200


    dir_path = os.path.dirname(os.path.realpath(__file__))
    rewards = np.load(dir_path + '/taxi-q/rewards.npy')
    for i in range(0, 30):
        print(i)
        agent = Q_learning(n_state, n_action, 0.005, 0.99)
        agent.Q = np.load(dir_path + '/taxi-q/q{}.npy'.format(i))
        SAS, f, avr_reward = roll_out(n_state, env, agent.get_pi(2.0), num_trajectory, truncate_size)
        rewards[i] = avr_reward
        np.save(dir_path + '/taxi-q/rewards.npy', rewards)