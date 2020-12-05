import numpy as np
import os
import matplotlib.pyplot as plt
import random

class BVFT(object):
    def __init__(self, data, gamma, rmax, rmin, resolution=1e-2, tabular=False):
        self.data = data
        self.n = len(data)
        self.rmax = rmax
        self.gamma = gamma
        self.vmax = rmax / (1.0 - gamma)
        self.vmin = rmin / (1.0 - gamma)
        self.res = resolution
        self.qs = None
        self.qs_discrete = None
        self.q_size = 0
        self.tabular = tabular


    def discretize(self, Q):
        if self.tabular:
            q_out = np.array([Q[t[0], t[1]] for t in self.data])
        else:
            q_out = np.array([Q.predict(t[0])[int(t[1])] for t in range(self.n)])

        discretized_q = np.digitize(q_out, np.linspace(self.vmin, self.vmax, int((self.vmax - self.vmin) / self.res) + 1), right=True)
        q_to_data_map = {}
        for i, q_val in enumerate(discretized_q):
            if q_val not in q_to_data_map:
                q_to_data_map[q_val] = i
            else:
                if isinstance(q_to_data_map[q_val], int):
                    q_to_data_map[q_val] = [q_to_data_map[q_val]]
                q_to_data_map[q_val].append(i)
        return q_out, discretized_q, q_to_data_map

    def discretize_qs(self, qs):
        self.qs = qs
        self.qs_discrete = [self.discretize(q) for q in qs]
        self.q_size = len(qs)

    def get_groups(self, q1_discretized, q2_discretized):
        q1_out, q1_inds, q1_dic = q1_discretized
        q2_out, q2_inds, q2_dic = q2_discretized
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

    def compute_Tf(self, Q1, groups):
        r = np.array([self.data[i][2] for i in range(self.n)])
        vfsp = [0 if self.data[i][3] is None else self.gamma * np.max(Q1[self.data[i][3]]) for i in range(self.n)]
        Tf = r + vfsp
        for group in groups:
            Tf[group] = np.sum(Tf[group]) / len(group)
        return Tf

    def compute_loss(self, q1_discretized, Tf):
        q1_out, q1_inds, q1_dic = q1_discretized
        diff = q1_out - Tf
        return np.sqrt(np.sum(diff**2))

    def get_loss(self, q1, q2):
        groups = self.get_groups(self.qs_discrete[q1], self.qs_discrete[q2])
        Tf1 = self.compute_Tf(self.qs[q1], groups)
        Tf2 = self.compute_Tf(self.qs[q2], groups)
        l1 = self.compute_loss(self.qs_discrete[q1], Tf1)
        l2 = self.compute_loss(self.qs_discrete[q2], Tf2)

        return l1, l2

    def get_loss_1(self, q1):
        groups = self.get_groups(self.qs_discrete[q1], self.qs_discrete[q1])
        Tf1 = self.compute_Tf(self.qs[q1], groups)
        l1 = self.compute_loss(self.qs_discrete[q1], Tf1)
        return l1

    def run(self, Qs, resolution=1e-2):
        self.res = resolution
        print("Being discretizing outputs of Q functions on batch data")
        self.discretize_qs(Qs)
        print("Starting pairwise comparison")

        loss_matrix = np.zeros((self.q_size, self.q_size))
        for i in range(self.q_size):
            for j in range(i, self.q_size):
                if i == j:
                    loss_matrix[i, i] = self.get_loss_1(i)
                    print("loss |Q{}; Q{}| = {}".format(i, i, loss_matrix[i, i]))
                else:
                    l1, l2 = self.get_loss(i, j)
                    loss_matrix[i, j] = l1
                    loss_matrix[j, i] = l2
                    print("loss |Q{}; Q{}| = {}".format(i, j, loss_matrix[i, j]))
                    print("loss |Q{}; Q{}| = {}".format(j, i, loss_matrix[j, i]))

        q_ranks = np.argsort(np.max(loss_matrix, axis=1))

        print("Ranking of Q functions:")
        print(q_ranks)

        return q_ranks, loss_matrix



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

    b = BVFT(test_data, gamma, rmax, rmin, tabular=True)
    b.run([Q1, Q2])



