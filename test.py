import gym
import d4rl
import numpy as np

# Create the environment
env = gym.make('minigrid-fourrooms-v0')
print(env.observation_space.shape)
print(env.action_space.n)

# d4rl abides by the OpenAI gym interface
env.reset()
print(env.step(env.action_space.sample()))

# # Each task is associated with a dataset
# # dataset contains observations, actions, rewards, terminals, and infos
# dataset = env.get_dataset()
# print(dataset['observations']) # An N x dim_observation Numpy array of observations
#
# # Alternatively, use d4rl.qlearning_dataset which
# # also adds next_observations.
dataset = d4rl.qlearning_dataset(env)
print(np.shape(dataset['observations'][0]))
print(dataset['actions'][:10])
print(dataset.keys())