import gym
import numpy as np
import os
from random_environment import TaxiRandom

rand = 0.5
env = TaxiRandom(5, rand)
dir_path = os.path.dirname(os.path.realpath(__file__))
num_states = env.n_state
num_actions = env.n_action
max_iterations = 1e5
delta = 10**-6

transition_iteration = 1e5
transition_epsilon = 1e-3
T, R = env.get_T_R(1e5, convergence=transition_epsilon)
np.save(dir_path + F'/../../data/taxi-random-{rand}/ENV_T_epsilon_{transition_epsilon}.npy', T)
np.save(dir_path + F'/../../data/taxi-random-{rand}/ENV_R_epsilon_{transition_epsilon}.npy', R)

T = np.load(dir_path + F'/../../data/taxi-random-{rand}/ENV_T_epsilon_{transition_epsilon}.npy')
V = np.zeros([num_states])
Q = np.zeros([num_states, num_actions])
dones = []
gamma = 0.99

print("Taxi")
print("Actions: ", num_actions)
print("States: ", num_states)

value_fn = np.zeros([num_states])

for i in range(int(max_iterations)):
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
np.save(dir_path + F'/../../data/taxi-random-{rand}/OPTIMAL_Q_GAMMA_{gamma}.npy', Q)


print("Value Iteration")
print("Iterations: ", iters)