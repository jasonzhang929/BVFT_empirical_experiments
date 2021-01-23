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
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

MODEL_NAME = "mountaincar-0.5"

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
                 epsilon_decay_steps=50000,
                 train_start=1000,
                 epsilon=1.0,
                 epsilon_min=0.02,
                 layers=None,
                 activation="relu",
                 lr=0.00025,
                 total_timesteps=100000,
                 max_mem_size=50000,
                 train_freq = 1,
                 rand=0.0):
        self.name = name
        self.env = gym.make('MountainCar-v0').env
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
        self.target_update_freq = 200
        self.lr = lr
        self.rand = rand
        self.train_freq = train_freq

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
            return np.argmax(self.model.predict(np.stack([state])))

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        state, action, reward, next_state, done = self.sample(self.batch_size)

        # do batch prediction to save speed
        target = self.target.predict(state)
        target_next = self.target.predict(next_state)
        target_next = reward + np.ndarray.flatten(self.gamma * (np.max(target_next, axis=-1)))
        target_next[done] = reward[done]
        target[np.arange(len(target_next)), action] = target_next

        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)

    def load(self, name):
        self.model = load_model(F"../data/{MODEL_NAME}/{name}")

    def save(self, name):
        self.model.save(F"../data/{MODEL_NAME}/{name}")

    def env_act_wrapper(self, action):
        if self.rand == 0.0 or np.random.random() > self.rand:
            return action
        else:
            return random.randrange(self.action_size)

    def run(self, save_points=None):
        if save_points is None:
            save_points = set([i * 2500 for i in range(2, 21)])

        print_every = 1000
        start_time = time.time()
        state = self.env.reset()
        episodes = 0
        cum_rewards = 0.0
        last_episode_rewards = 0.0
        next_state, done = None, None
        for t in range(self.total_timesteps):
            action = self.act(state)
            sum_rew = 0.0
            real_action = self.env_act_wrapper(action)
            for i in range(4):
                next_state, reward, done, _ = self.env.step(real_action)
                sum_rew += reward
            self.remember(state, action, sum_rew, next_state, done)
            state = next_state
            cum_rewards += sum_rew
            if done:
                state = self.env.reset()
                episodes += 1
                last_episode_rewards = cum_rewards
                cum_rewards = 0.0

            if t % self.train_freq == 0:
                self.replay()

            if t > self.train_start and t % self.target_update_freq == 0:
                self.target.set_weights(self.model.get_weights())

            if t in save_points:
                save_name = self.name + str(t) + F"_{np.random.randint(100000)}"
                print(F"Saving trained model as {save_name}.h5")
                self.save(F"{save_name}.h5")
            if t % print_every == 0:
                print("timesteps: {}, episode: {}, score: {}, e: {:.2}, time: {:.1}".format(t, episodes,
                     last_episode_rewards, self.epsilon, (time.time() - start_time)/60))


def generate_models():
    name_prefix = F"{MODEL_NAME}_DQN_"
    layers = [[256, 128, 64], [64, 64]]
    activations = ["relu", "tanh"]
    lrs = [0.00025, 0.0005]

    for lr in lrs:
        for layer in layers:
            layer_string = "_".join([str(i) for i in layer])
            for activation in activations:
                name = name_prefix + F"{layer_string}_{activation}_{lr}_"
                print(name)
                agent = DQNAgent(name,
                 epsilon_min=0.1,
                 layers=layer,
                 activation=activation,
                                 epsilon_decay_steps=30000,
                                 total_timesteps=50000,
                 lr=lr,
                                 rand=0.5)
                agent.run()



if __name__ == "__main__":
    generate_models()