import numpy as np
from BvftUtil import BvftRecord


class BVFT(object):
    def __init__(self, q_functions, data, gamma, rmax, rmin, record: BvftRecord = BvftRecord(), tabular=False,
                 verbose=False, bins=None,
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
        self.record = record

        if self.verbose:
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
        self.record.avg_q = [np.sum(qsa) for qsa in self.q_sa]

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
        # return np.sqrt(np.mean(diff ** 2))

    def get_bins(self, groups):
        group_sizes = [len(g) for g in groups]
        bin_ind = np.digitize(group_sizes, self.bins, right=True)
        percent_bins = np.zeros(len(self.bins) + 1)
        count_bins = np.zeros(len(self.bins) + 1)
        for i in range(len(group_sizes)):
            count_bins[bin_ind[i] + 1] += 1
            percent_bins[bin_ind[i] + 1] += group_sizes[i]
        percent_bins[0] = self.n - np.sum(percent_bins)
        count_bins[0] = percent_bins[0]
        return percent_bins, count_bins

    def run(self, resolution=1e-2):
        self.res = resolution
        if self.verbose:
            print(F"Being discretizing outputs of Q functions on batch data with resolution = {resolution}")
        self.discretize()
        if self.verbose:
            print("Starting pairwise comparison")
        percent_histos = []
        count_histos = []
        group_count = []
        loss_matrix = np.zeros((self.q_size, self.q_size))
        for q1 in range(self.q_size):
            for q2 in range(q1, self.q_size):
                groups = self.get_groups(q1, q2)
                percent_bins, count_bins = self.get_bins(groups)
                percent_histos.append(percent_bins)
                count_histos.append(count_bins)
                group_count.append(len(groups))

                loss_matrix[q1, q2] = self.compute_loss(q1, groups)
                # if self.verbose:
                #     print("loss |Q{}; Q{}| = {}".format(q1, q2, loss_matrix[q1, q2]))

                if q1 != q2:
                    loss_matrix[q2, q1] = self.compute_loss(q2, groups)
                    # if self.verbose:
                    #     print("loss |Q{}; Q{}| = {}".format(q2, q1, loss_matrix[q2, q1]))

        average_percent_bins = np.mean(np.array(percent_histos), axis=0) / self.n
        average_count_bins = np.mean(np.array(count_histos), axis=0)
        average_group_count = np.mean(group_count)
        if self.verbose:
            print(np.max(loss_matrix, axis=1))
        self.record.resolutions.append(resolution)
        self.record.losses.append(np.max(loss_matrix, axis=1))
        self.record.loss_matrices.append(loss_matrix)
        self.record.percent_bin_histogram.append(average_percent_bins)
        self.record.count_bin_histogram.append(average_count_bins)
        self.record.group_counts.append(average_group_count)


    def compute_optimal_group_skyline(self):
        groups = self.get_groups(self.q_size-1, self.q_size-1)
        loss = [self.compute_loss(q, groups) for q in range(self.q_size)]
        self.record.optimal_grouping_skyline.append(np.array(loss))


    def get_br_ranking(self):
        br = [np.sqrt(np.sum((self.q_sa[q] - self.r_plus_vfsp[q]) ** 2)) for q in range(self.q_size)]
        br_rank = np.argsort(br)
        self.record.bellman_residual = br
        return br_rank


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
    record = BvftRecord()
    b = BVFT([Q1, Q2], test_data, gamma, rmax, rmin, record, tabular=True, verbose=True)
    b.run()
