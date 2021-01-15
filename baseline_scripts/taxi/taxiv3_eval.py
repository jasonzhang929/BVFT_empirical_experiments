from os import listdir
from os.path import isfile, join
import os, time, random, pickle
import numpy as np
import gym


def roll_out(env, Q, gamma, num_trajectory, truncate_size, verbose=False):
    total_reward = 0.0
    actions = np.argmax(Q, axis=-1)
    for i_trajectory in range(num_trajectory):
        state = env.reset()
        r = 0.0
        g = 1.0
        for i_t in range(truncate_size):
            action = actions[state]
            state, reward, done, info = env.step(action)
            r += g * reward
            g *= gamma
            if done:
                break
        total_reward += r
        if verbose and i_trajectory % 1000 == 0:
            print(i_trajectory, total_reward * (1.0 - gamma)/(i_trajectory + 1))
    return total_reward * (1.0 - gamma) / num_trajectory


def eval_directory(folder_name, gamma, episodes=5000, truncate_size=100):
    dir_path = os.path.dirname(os.path.realpath(__file__)) + F"/../../data/{folder_name}/"
    onlyfiles = set([f for f in listdir(dir_path) if isfile(join(dir_path, f))])
    models_to_eval = []
    for file in onlyfiles:
        if "DATA" not in file and ".npy" in file:
            if "VALUE" in file:
                continue
            models_to_eval.append(file)

    print(F"{len(models_to_eval)} models to be evaluated")
    for model_name in models_to_eval:
        model_path = dir_path + model_name
        env = gym.make('Taxi-v3')
        print(F"evaluating model {model_name}")
        Q = np.load(model_path)
        start = time.time()
        value = roll_out(env, Q, gamma, episodes, truncate_size)
        new_name = "{}_VALUE_{:.7}.npy".format(model_name[:-4], value)
        new_path = dir_path + new_name
        os.rename(model_path, new_path)
        print(F"renamed {model_name} as {new_name}, took {time.time() - start} seconds")


def generate_data(env, Q, data_size, epsilon, truncate_size=100):
    data = []
    actions = np.argmax(Q, axis=-1)
    n_action = env.action_space.n
    ts = 0
    while ts < data_size:
        state = env.reset()
        for i in range(truncate_size):
            if np.random.rand() > epsilon:
                action = actions[state]
            else:
                action = np.random.randint(n_action)
            next_state, reward, done, info = env.step(action)
            ts += 1
            if done:
                data.append((state, action, reward, None, done))
                break
            else:
                data.append((state, action, reward, next_state, done))
                state = next_state
    return data


def generate_datasets(folder_name, epsilons, data_size=1e5, count=1, threshold=1.0):
    dir_path = os.path.dirname(os.path.realpath(__file__)) + F"/../../data/{folder_name}/"
    onlyfiles = set([f for f in listdir(dir_path) if isfile(join(dir_path, f))])
    models_to_eval = []
    for file in onlyfiles:
        if "VALUE" in file and float(file.split('_')[-1][:-4]) > threshold:
            models_to_eval.append(file)
    for epsilon in epsilons:
        for model_name in random.sample(models_to_eval, count, ):
            model_path = dir_path + model_name
            env = gym.make('Taxi-v3')
            print(F"generate data with model {model_name} and epsilon{epsilon}")
            Q = np.load(model_path)
            start = time.time()
            data = generate_data(env, Q, data_size, epsilon)
            data_name = "DATA_{}_EPS_{:.4}".format(np.random.randint(1e5), epsilon)

            # np.save(dir_path + data_name, np.array(data, dtype=int))
            outfile = open(dir_path + data_name, "wb")
            pickle.dump(data, outfile)
            outfile.close()
            print(F"saved to {data_name}, took {time.time() - start} seconds")


if __name__ == '__main__':
    # eval_directory("taxi-v3", 0.95)
    epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    generate_datasets("taxi-v3", epsilons, threshold=0.08)
