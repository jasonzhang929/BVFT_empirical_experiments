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

    model = Model(inputs = X_input, outputs = X, name='CartPole_DQN_model')
    model.compile(loss="mse", optimizer=RMSprop(lr=lr, rho=0.95, epsilon=0.01), metrics=["accuracy"])

    model.summary()
    return model

class DQNAgent:

    def __init__(self, name,
                 batch_size=64,
                 gamma=0.99,
                 epsilon_decay=0.999,
                 train_start=1000,
                 epsilon=1.0,
                 epsilon_min=0.001,
                 layers=None,
                 activation="relu"):
        self.name = name
        self.env = gym.make('CartPole-v1')
        # by default, CartPole-v1 has max episode steps = 500
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.EPISODES = 1000
        self.memory = deque(maxlen=50000)
        
        self.gamma = gamma    # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.train_start = train_start
        self.target_update_freq = 500

        # create main model
        self.model = OurModel(input_shape=(self.state_size,), action_space=self.action_size,
                              layers=layers, activation=activation)
        self.target = OurModel(input_shape=(self.state_size,), action_space=self.action_size,
                              layers=layers, activation=activation)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state))

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        # Randomly sample minibatch from the memory
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # do batch prediction to save speed
        target = self.target.predict(state)
        target_next = self.target.predict(next_state)

        for i in range(self.batch_size):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # Standard - DQN
                # DQN chooses the max Q value among next actions
                # selection and evaluation of action is on the target Q Network
                # Q_max = max_a' Q_target(s', a')
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)


    def load(self, name):
        self.model = load_model(name)

    def save(self, name):
        self.model.save("../data/cartpole/" + name)

    def save_data(self, name):
        outfile = open("../data/cartpole/data_" + name, "wb")
        pickle.dump(self.memory, outfile)
        outfile.close()
            
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
                self.env.render()
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
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
                        save_name = self.name + str(save_point)
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
                self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, _ = self.env.step(action)
                state = np.reshape(next_state, [1, self.state_size])
                i += 1
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, i))
                    break




def generate_models():
    name_prefix = "cartpole_DQN_"
    layers = [[512, 256, 64], [256, 128, 64], [64, 64]]
    epsilon_decays = [0.999, 0.99995]
    activations = ["relu", "tanh"]


    for layer in layers:
        layer_string = "_".join([str(i) for i in layer])
        for epsilon_decay in epsilon_decays:
            for activation in activations:
                name = name_prefix + F"{layer_string}_{epsilon_decay}_{activation}_"
                agent = DQNAgent(name, batch_size=64,
                 epsilon_decay=epsilon_decay,
                 epsilon_min=0.02,
                 layers=layer,
                 activation=activation)
                agent.run()





if __name__ == "__main__":
    agent = DQNAgent("test")
    agent.run()
    # agent.test()

    # generate_models()
    # file = open("../data/cartpole/data_teststart", 'rb')
    # d = pickle.load(file)
    # print(d)