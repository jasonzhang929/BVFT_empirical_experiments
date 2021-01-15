import gym
import numpy as np
import os

env = gym.make('FrozenLake-v0', map_name='8x8')
dir_path = os.path.dirname(os.path.realpath(__file__))
num_states = env.observation_space.n
num_actions = env.action_space.n
max_iterations = 2000
delta = 10**-6

R = np.zeros([num_states, num_actions, num_states])
T = np.zeros([num_states, num_actions, num_states])
V = np.zeros([env.observation_space.n])
Q = np.zeros([env.observation_space.n, env.action_space.n])
dones = []
gamma = 0.95

print("FrozenLake-v0")
print("Actions: ", num_actions)
print("States: ", num_states)
print(env.env.desc)

for state in range(num_states):
    for action in range(num_actions):
        for transition in env.env.P[state][action]:
            probability, next_state, reward, done = transition
            R[state, action, next_state] = reward
            T[state, action, next_state] = probability
            if done:
                dones.append((state, action))
        T[state, action, :] /= np.sum(T[state, action, :])

value_fn = np.zeros([num_states])

for i in range(max_iterations):
    previous_value_fn = value_fn.copy()
    #learn more about einsum
    Q = np.einsum('ijk,ijk -> ij', T, R + gamma * value_fn)
    for d in dones:
        Q[d[0], d[1]] = 20
    value_fn = np.max(Q, axis=1)
    loss = np.max(np.abs(value_fn - previous_value_fn))
    if loss < delta:
        break
    policy = np.argmax(Q, axis=1)
    if i % 10 == 0:
        print(F"iteration {i + 1}, loss = {loss}")
iters = i + 1
np.save(dir_path + F'/../../data/frozenlake/OPTIMAL_Q_GAMMA_{gamma}.npy', Q)


print("Value Iteration")
print("Iterations: ", iters)