import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # -1:cpu, 0:first gpu
import gym
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution() # usually using this for fastest performance
from tensorflow.keras.models import load_model
from os import listdir
from os.path import isfile, join
from multiprocessing import Process, Pipe
from collections import deque
import time, pickle, random
from MonteCarlo_eval_pendulum import Pendulum2DEnv

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    print(f'GPUs {gpus}')
    try: tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError: pass


class Environment(Process):
    def __init__(self, env_idx, child_conn, env_name, state_size, action_size, visualize=False):
        super(Environment, self).__init__()
        self.env = Pendulum2DEnv()
        self.is_render = visualize
        self.env_idx = env_idx
        self.child_conn = child_conn
        self.state_size = state_size
        self.action_size = action_size

    def run(self):
        super(Environment, self).run()
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size])
        self.child_conn.send(state)
        while True:
            action = self.child_conn.recv()
            if self.is_render and self.env_idx == 0:
                self.env.render()

            state, reward, done, info = self.env.step(action)
            state = np.reshape(state, [1, self.state_size])

            if done:
                state = self.env.reset()
                state = np.reshape(state, [1, self.state_size])

            self.child_conn.send([state, reward, done, info])

class DataGenAgent():
    def __init__(self, env_name, model_path, epsilon=0.2, time_steps=50000):
        # Initialization
        # Environment and PPO parameters
        self.env_name = env_name
        self.env = Pendulum2DEnv()
        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.shape
        self.TOTAL_TIME = time_steps
        self.epsilon = epsilon
        self.memory = deque(maxlen=time_steps)
        self.model = load_model(model_path)

    def generate_data(self, num_worker=4):
        works, parent_conns, child_conns = [], [], []
        for idx in range(num_worker):
            parent_conn, child_conn = Pipe()
            work = Environment(idx, child_conn, self.env_name, self.state_size[0], self.action_size, False)
            work.start()
            works.append(work)
            parent_conns.append(parent_conn)
            child_conns.append(child_conn)

        state = [0 for _ in range(num_worker)]
        epi_length = [0 for _ in range(num_worker)]
        for worker_id, parent_conn in enumerate(parent_conns):
            state[worker_id] = parent_conn.recv()
        start_time = time.time()
        print_every = 5000

        while len(self.memory) < self.TOTAL_TIME:
            q_value_list = self.model.predict(np.reshape(state, [num_worker, self.state_size[0]]))
            actions_list = np.argmax(q_value_list, axis=-1)

            for worker_id, parent_conn in enumerate(parent_conns):
                if np.random.random() <= self.epsilon:
                    actions_list[worker_id] = np.random.randint(self.action_size)
                parent_conn.send(actions_list[worker_id])
                # actions[worker_id].append(actions_list[worker_id])

            for worker_id, parent_conn in enumerate(parent_conns):
                next_state, reward, done, _ = parent_conn.recv()

                # epi_length[worker_id] += 1
                # if not done or epi_length[worker_id] == self.env._max_episode_steps:
                #     reward = reward
                # else:
                #     reward = -reward

                state[worker_id] = next_state
                data = (state[worker_id], actions_list[worker_id], reward, next_state, done)
                self.memory.append(data)

                # if done:
                #     epi_length[worker_id] = 0

                # if len(self.memory) % print_every == 0:
                #     print(
                #         F"TimeSteps: {len(self.memory)}, per {print_every} time: {print_every * (time.time() - start_time) / len(self.memory)}")
                #

        # terminating processes after while loop
        works.append(work)
        for work in works:
            work.terminate()
            # print('TERMINATED:', work)
            work.join()

    def save_data(self, path):
        outfile = open(path, "wb")
        pickle.dump(self.memory, outfile)
        outfile.close()


def gen_data_directory(env_name, folder_name, threshold, num_data=10, data_size=50000, num_worker=8, epsilon=0.2):
    dir_path = F"../data/{folder_name}/"
    onlyfiles = set([f for f in listdir(dir_path) if isfile(join(dir_path, f))])
    models_to_gen = []
    for file in onlyfiles:
        if "VALUE" in file:
            value = float(file.split('_')[-1][:-3])
            if value > threshold:
                models_to_gen.append((file, value))
    print(F"{len(models_to_gen)} models available for data generation")

    for model_name, value in random.sample(models_to_gen, num_data):
        model_path = F"../data/{folder_name}/{model_name}"

        data_gen = DataGenAgent(env_name, model_path, time_steps=data_size, epsilon=epsilon)

        data_gen.generate_data(num_worker=num_worker)

        new_name = "DATA_{}_EPS_{:.5}.h5".format(np.random.randint(10**6), epsilon)
        new_path = F"../data/{folder_name}/{new_name}"
        data_gen.save_data(new_path)
        print(F"Data saved at {new_name}")


if __name__ == "__main__":
    env_name = ""
    folder_name = 'pendulum'

    threshold = -180
    num_data = 9
    data_size = 100000
    epsilons = [0.1, 0.5, 1.0]
    num_worker = 4
    for epsilon in epsilons:
        gen_data_directory(env_name, folder_name, threshold, num_data=num_data, data_size=data_size, num_worker=num_worker,
                       epsilon=epsilon)
