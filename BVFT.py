import numpy as np
import os
import matplotlib.pyplot as plt
import random
from collections import Counter


class BVFT(object):
    def __init__(self, q_functions, data, gamma, rmax, rmin, tabular=False, verbose=False, bins=None,
                 profiling=False):
        self.data = data
        self.n = len(data)
        self.rmax = rmax
        self.gamma = gamma
        self.vmax = rmax / (1.0 - gamma)
        self.vmin = rmin / (1.0 - gamma)
        self.res = 0
        self.q_sa_discrete = []
        self.q_to_data_map = []
        self.q_size = len(q_functions)
        self.verbose = verbose
        if bins is None:
            bins = [2, 3, 4, 8, 16, 100, 1e10]
        self.bins = bins
        self.q_sa = []
        self.r_plus_vfsp = []
        self.q_functions = q_functions

        print(F"Data size = {self.n}")
        rewards = np.array([t[2] for t in self.data])

        actions = [int(t[1]) for t in self.data]

        if tabular:
            states = np.array([t[0] for t in self.data])
            for Q in q_functions:
                self.q_sa.append(np.array([Q[states[i], actions[i]] for i in range(self.n)]))
                vfsp = np.array([0.0 if t[3] is None else np.max(Q[t[3]]) for t in self.data])
                self.r_plus_vfsp.append(rewards + self.gamma * vfsp)
        else:
            next_states = np.array([t[3][0] for t in self.data])
            states = np.array([t[0][0] for t in self.data])
            for Q in q_functions:
                qs = Q.predict(states)
                # print(qs)
                self.q_sa.append(np.array([qs[i][actions[i]] for i in range(self.n)]))
                vfsp = np.max(Q.predict(next_states), axis=1)
                self.r_plus_vfsp.append(rewards + self.gamma * vfsp)

    def discretize(self):
        self.q_sa_discrete = []
        self.q_to_data_map = []
        bins = int((self.vmax - self.vmin) / self.res) + 1

        for q in self.q_sa:
            discretized_q = np.digitize(q, np.linspace(self.vmin, self.vmax, bins), right=True)
            self.q_sa_discrete.append(discretized_q)
            q_to_data_map = {}
            for i, q_val in enumerate(discretized_q):
                if q_val not in q_to_data_map:
                    q_to_data_map[q_val] = i
                else:
                    if isinstance(q_to_data_map[q_val], int):
                        q_to_data_map[q_val] = [q_to_data_map[q_val]]
                    q_to_data_map[q_val].append(i)
            self.q_to_data_map.append(q_to_data_map)

    def get_groups(self, q1, q2):
        q1_dic = self.q_to_data_map[q1]
        q2_inds, q2_dic = self.q_sa_discrete[q2], self.q_to_data_map[q2]
        groups = []
        for key in q1_dic:
            if isinstance(q1_dic[key], list):
                q1_list = q1_dic[key]
                set1 = set(q1_list)
                for p1 in q1_list:
                    if p1 in set1 and isinstance(q2_dic[q2_inds[p1]], list):
                        set2 = set(q2_dic[q2_inds[p1]])
                        intersect = set1.intersection(set2)
                        set1 = set1.difference(intersect)
                        if len(intersect) > 1:
                            groups.append(list(intersect))
        return groups

    def compute_loss(self, q1, groups):
        Tf = self.r_plus_vfsp[q1].copy()
        for group in groups:
            Tf[group] = np.mean(Tf[group])
        diff = self.q_sa[q1] - Tf
        return np.sqrt(np.sum(diff ** 2))

    def get_bins(self, groups):
        group_sizes = [len(g) for g in groups]
        bin_ind = np.digitize(group_sizes, self.bins, right=True)
        bins = np.zeros(len(self.bins) + 1)
        for i in range(len(group_sizes)):
            bins[bin_ind[i] + 1] += group_sizes[i]
        bins[0] = self.n - np.sum(bins)
        return bins

    def run(self, resolution=1e-2):
        self.res = resolution
        print(F"Being discretizing outputs of Q functions on batch data with resolution = {resolution}")
        self.discretize()
        print("Starting pairwise comparison")
        histos = []
        loss_matrix = np.zeros((self.q_size, self.q_size))
        for q1 in range(self.q_size):
            for q2 in range(q1, self.q_size):
                groups = self.get_groups(q1, q2)
                histos.append(self.get_bins(groups))

                loss_matrix[q1, q2] = self.compute_loss(q1, groups)
                if self.verbose:
                    print("loss |Q{}; Q{}| = {}".format(q1, q2, loss_matrix[q1, q2]))

                if q1 != q2:
                    loss_matrix[q2, q1] = self.compute_loss(q2, groups)
                    if self.verbose:
                        print("loss |Q{}; Q{}| = {}".format(q2, q1, loss_matrix[q2, q1]))

        q_ranks = np.argsort(np.max(loss_matrix, axis=1))
        bin_histo = np.mean(histos, axis=0)/self.n

        print("Ranking of Q functions:")
        print(q_ranks)

        return q_ranks, loss_matrix, bin_histo

    def get_br_ranking(self):
        br = [np.sqrt(np.sum((self.q_sa[q] - self.r_plus_vfsp[q]) ** 2)) for q in range(self.q_size)]
        return np.argsort(br)

if __name__ == '__main__':
    # Toy example in the paper
    test_data = [(0, 0, 1.0, 1), (0, 1, 1.0, 2),
                 (1, 0, 0.0, None), (1, 1, 0.0, None),
                 (2, 0, 1.0, None), (2, 1, 1.0, None),
                 (3, 0, 1.0, 4), (3, 1, 1.0, 4)]

    Q1 = np.array([[1.0, 1.9], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    Q2 = np.array([[7.0, 1.9], [0.0, 0.0], [1.0, 1.0], [7.0, 7.0], [10.0, 10.0]])

    gamma = 0.9
    rmax, rmin = 1.0, 0.0

    b = BVFT([Q1, Q2], test_data, gamma, rmax, rmin, tabular=True, verbose=True)
    b.run()
