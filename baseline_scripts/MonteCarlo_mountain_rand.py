import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # -1:cpu, 0:first gpu
import gym
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()  # usually using this for fastest performance
from tensorflow.keras.models import Model, load_model
from os import listdir
from os.path import isfile, join
from multiprocessing import Process, Pipe
import time

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    print(f'GPUs {gpus}')
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError:
        pass


class Environment(Process):
    def __init__(self, env_idx, child_conn, env_name, state_size, action_size, random, visualize=False):
        super(Environment, self).__init__()
        self.env = gym.make(env_name).env
        self.is_render = visualize
        self.env_idx = env_idx
        self.child_conn = child_conn
        self.state_size = state_size
        self.action_size = action_size
        self.rand = random

    def run(self):
        super(Environment, self).run()
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size])
        self.child_conn.send(state)
        done = False
        while True:
            action = self.child_conn.recv()
            if np.random.random() < self.rand:
                action = np.random.randint(self.action_size)
            if self.is_render and self.env_idx == 0:
                self.env.render()
            total_reward = 0.0
            for i in range(4):
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                if done:
                    break
            state = np.reshape(state, [1, self.state_size])

            if done:
                state = self.env.reset()
                state = np.reshape(state, [1, self.state_size])

            self.child_conn.send([state, total_reward, done, None])


class RollOutAgent():
    def __init__(self, env_name, model_path, episodes=100):
        # Initialization
        # Environment and PPO parameters
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.shape
        self.EPISODES = episodes  # total episodes to train through all environments
        self.episode = 0  # used to track the episodes total count of episodes played through all thread environments
        self.model = load_model(model_path)

    def eval(self, num_worker=4):
        works, parent_conns, child_conns = [], [], []
        for idx in range(num_worker):
            parent_conn, child_conn = Pipe()
            work = Environment(idx, child_conn, self.env_name, self.state_size[0], self.action_size, rand, False)
            work.start()
            works.append(work)
            parent_conns.append(parent_conn)
            child_conns.append(child_conn)

        # states = [[] for _ in range(num_worker)]
        # next_states = [[] for _ in range(num_worker)]
        # actions = [[] for _ in range(num_worker)]
        # rewards = [[] for _ in range(num_worker)]
        # dones = [[] for _ in range(num_worker)]
        score = [0 for _ in range(num_worker)]
        scores = []
        epi_len = [0 for _ in range(num_worker)]

        state = [0 for _ in range(num_worker)]
        for worker_id, parent_conn in enumerate(parent_conns):
            state[worker_id] = parent_conn.recv()
        start_time = time.time()
        while self.episode < self.EPISODES:
            q_value_list = self.model.predict(np.reshape(state, [num_worker, self.state_size[0]]))
            actions_list = np.argmax(q_value_list, axis=-1)

            for worker_id, parent_conn in enumerate(parent_conns):
                parent_conn.send(actions_list[worker_id])
                # actions[worker_id].append(actions_list[worker_id])

            for worker_id, parent_conn in enumerate(parent_conns):
                next_state, reward, done, _ = parent_conn.recv()
                state[worker_id] = next_state
                score[worker_id] += reward
                epi_len[worker_id] += 1

                if done or epi_len[worker_id] == 250:
                    scores.append(score[worker_id])
                    score[worker_id] = 0
                    epi_len[worker_id] = 0

                    if (self.episode < self.EPISODES):
                        self.episode += 1
                    if self.episode % 50 == 0:
                        print(F"Episode: {self.episode}, reward mean: {np.mean(scores)}, "
                              F"per episode time: {(time.time() - start_time) / self.episode}")

        # terminating processes after while loop
        works.append(work)
        for work in works:
            work.terminate()
            # print('TERMINATED:', work)
            work.join()

        return np.mean(scores)

    def save(self, path):
        self.model.save(path)


def eval_directory(env_name, folder_name, episodes=100, num_worker=4):
    dir_path = F"../data/{folder_name}/"
    onlyfiles = set([f for f in listdir(dir_path) if isfile(join(dir_path, f))])
    models_to_eval = []
    for file in onlyfiles:
        if "VALUE" not in file and 'DATA' not in file:
            models_to_eval.append(file)
    print(F"{len(models_to_eval)} models to be evaluated")
    for model_name in models_to_eval:
        model_path = F"../data/{folder_name}/{model_name}"
        print(model_path)
        ev = RollOutAgent(env_name, model_path, episodes=episodes)
        value = ev.eval(num_worker=num_worker)
        new_name = "{}_VALUE_{:.5}.h5".format(model_name[:-3], value)
        new_path = F"../data/{folder_name}/{new_name}"
        os.rename(model_path, new_path)
        print(F"renamed {model_name} as {new_name}")

# def eval_directory(env_name, folder_name, episodes=100, num_worker=4):
#     dir_path = F"../data/{folder_name}/"
#     onlyfiles = set([f for f in listdir(dir_path) if isfile(join(dir_path, f))])
#     models_to_eval = []
#     for file in onlyfiles:
#         if 'DATA' not in file:
#             models_to_eval.append(file)
#     print(F"{len(models_to_eval)} models to be evaluated")
#     for model_name in models_to_eval:
#         model_path = F"../data/{folder_name}/{model_name}"
#         print(model_path)
#         ev = RollOutAgent(env_name, model_path, episodes=episodes)
#         value = ev.eval(num_worker=num_worker)
#         if "VALUE" in model_name:
#             model_name = "_".join(model_name.split("_")[:-2])
#         else:
#             model_name = model_name[:-3]
#         new_name = "{}_VALUE_{:.5}.h5".format(model_name, value)
#         new_path = F"../data/{folder_name}/{new_name}"
#         os.rename(model_path, new_path)
#         print(F"renamed {model_name} as {new_name}")


if __name__ == "__main__":
    rand = 0.5
    env_name = 'MountainCar-v0'
    folder_name = F'mountaincar-{rand}'

    eval_directory(env_name, folder_name, episodes=100)
