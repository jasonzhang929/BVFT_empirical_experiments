# Tutorial by www.pylessons.com
# Tutorial written for - Tensorflow 1.15, Keras 2.2.4

import os
from typing import Any

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import gym
import numpy as np
from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop
import pickle, time

MODEL_NAME = "LunarLander"

def OurModel(input_shape, action_space, layers=None, activation="relu", lr=0.00025):
    if layers is None:
        layers = [512, 256, 64]

    if len(layers) < 1:
        print("error layers cant be empty")
        return

    X_input = Input(input_shape)

    # 'Dense' is the basic form of a neural network layer
    # Input Layer of state size(4) and Hidden Layer with 512 nodes
    X = Dense(layers[0], input_shape=input_shape, activation=activation, kernel_initializer='he_uniform')(X_input)

    for size in layers[1:]:
        X = Dense(size, activation=activation, kernel_initializer='he_uniform')(X)

    # Output Layer with # of actions: 2 nodes (left, right)
    X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

    model = Model(inputs=X_input, outputs=X, name=F'{MODEL_NAME}_DQN_model')
    model.compile(loss="mse", optimizer=RMSprop(lr=lr, rho=0.95, epsilon=0.01), metrics=["accuracy"])

    model.summary()
    return model

class DQNAgent:

    def __init__(self, name,
                 batch_size=32,
                 gamma=0.99,
                 epsilon_decay_steps=20000,
                 train_start=1000,
                 epsilon=1.0,
                 epsilon_min=0.02,
                 layers=None,
                 activation="relu",
                 lr=0.00025,
                 total_timesteps=100000,
                 max_mem_size=50000,
                 rand=0.0):
        self.name = name
        self.env = gym.make('LunarLander-v2')
        self.total_timesteps = total_timesteps
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.memory = []
        self.max_mem_size = max_mem_size
        self.mem_next_index = 0
        self.gamma = gamma    # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.train_start = train_start
        self.target_update_freq = 500
        self.lr = lr
        self.rand = rand

        self.model = OurModel(input_shape=(self.state_size,), action_space=self.action_size,
                              layers=layers, activation=activation, lr=lr)
        self.target = OurModel(input_shape=(self.state_size,), action_space=self.action_size,
                              layers=layers, activation=activation, lr=lr)

    def remember(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        if self.mem_next_index >= len(self.memory):
            self.memory.append(data)
        else:
            self.memory[self.mem_next_index] = data
        self.mem_next_index = (self.mem_next_index + 1) % self.max_mem_size

        if len(self.memory) > self.train_start and self.epsilon > self.epsilon_min:
                self.epsilon -= (1.0 - self.epsilon_min)/self.epsilon_decay_steps

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self.memory[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self.memory) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state))

    def replay(self):
        if len(self.memory) < self.train_start:
            return

        state, action, reward, next_state, done = self.sample(self.batch_size)

        # do batch prediction to save speed
        target = self.target.predict(state)
        target_next = self.target.predict(next_state)
        target_next = reward + self.gamma * (np.max(target_next, axis=-1))
        target_next[done] = reward[done]
        target[np.arange(len(target_next)), action] = target_next

        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)


    def load(self, name):
        self.model = load_model("../data/cartpole/" + name)

    def save(self, name):
        self.model.save("../data/cartpole/" + name)

    def save_data(self, name):
        outfile = open("../data/cartpole/data_" + name, "wb")
        pickle.dump(self.memory, outfile)
        outfile.close()

    def env_act_wrapper(self, action):
        if self.rand == 0.0 or np.random.random() > self.rand:
            return action
        else:
            return random.randrange(self.action_size)

    def run(self):
        save_point = 200
        j = 0
        start = time.time()
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                # self.env.render()
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(self.env_act_wrapper(action))
                next_state = np.reshape(next_state, [1, self.state_size])
                if not done or i == self.env._max_episode_steps-1:
                    reward = reward
                else:
                    reward = -100
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                j += 1
                if done:
                    print("episode: {}/{}, score: {}, e: {:.2}, time: {:.1}".format(e, self.EPISODES, i, self.epsilon,
                                                                                    (time.time() - start)/60))
                    if i >= save_point:
                        cum_reward = self.roll_out_eval()
                        save_name = self.name + str(save_point) + "_{:.5f}".format(cum_reward)
                        print(F"Saving trained model as {save_name}.h5")
                        self.save(F"{save_name}.h5")
                        save_point += 100
                    if e == 80:
                        self.save_data(self.name + "start")
                    if i >= 500:
                        self.save_data(self.name + "end")
                        return
                self.replay()
                if j % self.target_update_freq == 0:
                    self.target.set_weights(self.model.get_weights())

    def test(self):
        self.load("cartpole-dqn.h5")
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                # self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, _ = self.env.step(action)
                state = np.reshape(next_state, [1, self.state_size])
                i += 1
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, i))
                    break

    def roll_out_eval(self):
        length = 500
        episodes = 50
        scores = np.zeros(episodes)
        print("begin eval")
        for e in range(episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            reward_sum = 0
            if e%10==0:
                print(e)
            while not done:
                # self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, _ = self.env.step(self.env_act_wrapper(action))
                state = np.reshape(next_state, [1, self.state_size])
                i += 1
                if done and i < self.env._max_episode_steps:
                    reward = -100
                reward_sum += reward
                if done or i == length:
                    break
            scores[e] = reward_sum
        mean_cum_reward = np.mean(scores)
        print("Eval result: =============================== average reward = {:.5f}".format(mean_cum_reward))
        return mean_cum_reward

    def generate_explorative_data(self, policy_name, size=100000):
        start = time.time()
        self.load(policy_name)
        while len(self.memory) < size:
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(self.env_act_wrapper(action))
                next_state = np.reshape(next_state, [1, self.state_size])
                if not done or i == self.env._max_episode_steps - 1:
                    reward = reward
                else:
                    reward = -100
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
            if len(self.memory) % 10000 == 0:
                print(F"{len(self.memory)} time steps, {(time.time() - start)*1e4/60/len(self.memory)} min per 10000 time step")
        self.save_data(self.name)

def generate_data():
    policy_name = "test500_395.99000.h5"
    epsilon_mins = [0.2, 0.4, 0.6]
    for epsilon_min in epsilon_mins:
        for i in range(10):
            model_name = F"Explore_data_cartpole_DQN_{epsilon_mins}_{i}"
            agent = DQNAgent(model_name, epsilon_min=epsilon_min)
            agent.generate_explorative_data(policy_name)


def generate_models():
    name_prefix = "cartpole_DQN_"
    layers = [[256, 128, 64], [64, 64]]
    epsilon_decays = [0.999]
    activations = ["relu", "tanh"]
    lrs = [0.00025, 0.0005]

    for lr in lrs:
        for layer in layers:
            layer_string = "_".join([str(i) for i in layer])
            for epsilon_decay in epsilon_decays:
                for activation in activations:
                    name = name_prefix + F"{layer_string}_{epsilon_decay}_{activation}_{lr}_"
                    print(name)
                    agent = DQNAgent(name, batch_size=64,
                     epsilon_decay=epsilon_decay,
                     epsilon_min=0.02,
                     layers=layer,
                     activation=activation,
                     lr=lr)
                    agent.run()


def generate_random_models():

    layers = [[256, 128, 64], [512, 64]]
    epsilon_decays = [0.999]
    activations = ["relu", "tanh"]
    lrs = [0.00025, 0.0005]
    rands = [0.2, 0.4, 0.6]
    for rand in rands:
        name_prefix = F"cartpole_RAND{int(rand*100)}_DQN_"
        for lr in lrs:
            for layer in layers:
                layer_string = "_".join([str(i) for i in layer])
                for epsilon_decay in epsilon_decays:
                    for activation in activations:
                        name = name_prefix + F"{layer_string}_{epsilon_decay}_{activation}_{lr}_"
                        print(name)
                        agent = DQNAgent(name, batch_size=64,
                         epsilon_decay=epsilon_decay,
                         epsilon_min=0.02,
                         layers=layer,
                         activation=activation,
                         lr=lr, rand=rand)
                        agent.run()


if __name__ == "__main__":
    # agent = DQNAgent("test")
    # agent.roll_out_eval()
    # agent.run()
    # agent.test()

    # generate_models()
    # generate_random_models()
    generate_data()
    # file = open("../data/cartpole/data_teststart", 'rb')
    # d = pickle.load(file)
    # print(d)