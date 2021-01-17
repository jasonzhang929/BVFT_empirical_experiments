import matplotlib.pyplot as plt
import numpy as np
import pickle, os, random
from scipy.stats import pearsonr, sem
import gym
from collections import OrderedDict


class BvftRecord(object):
    def __init__(self,
                 data_size=None,
                 data_explore_rate=None,
                 gamma=None,
                 model_count=None):
        self.data_size = data_size
        self.data_explore_rate = data_explore_rate
        self.gamma = gamma
        self.model_count = model_count
        self.resolutions = []
        self.group_counts = []
        self.losses = []
        self.loss_matrices = []
        self.model_values = None
        self.percent_bin_histogram = []
        self.count_bin_histogram = []
        self.bellman_residual = None
        self.bellman_error = None
        self.q_names = None
        self.q_star_diff = None
        self.avg_q = None
        self.optimal_grouping_skyline = []
        self.hyper_parameters = [self.data_size,
                                 self.data_explore_rate,
                                 self.gamma,
                                 self.model_count]

    def match(self, other):
        other_hyper = other.hyper_parameters
        for i, para in enumerate(self.hyper_parameters):
            if para is not None and para != other_hyper[i]:
                return False
        return True


def plot_performance_vs_rank_bar(axs, record: BvftRecord, resolution):
    if resolution not in record.resolutions:
        print("resolution not found in bvft record!")
        return
    index = record.resolutions.index(resolution)
    ranks = np.argsort(record.losses[index])
    axs.bar([i + 1 for i in range(record.model_count)], record.model_values[ranks] - np.mean(record.model_values))
    axs.set_title("samples = {}, resolution = {:.5f}".format(record.data_size, resolution))
    axs.set_ylabel('Actual value advantage')


def plot_percent_bin_sizes(axs, record: BvftRecord, resolution, bins, plot_loc=None):
    if plot_loc is None:
        plot_loc = (False, False, False, False)
    top, bot, left, right = plot_loc
    if resolution not in record.resolutions:
        print("resolution not found in bvft record!")
        return
    index = record.resolutions.index(resolution)
    axs.bar([str(i) for i in ([1] + bins)], record.percent_bin_histogram[index])
    title_text = "|D| = {}, res = {:.3f}, {:.4} groups avg.".format(record.data_size, resolution,
                                                                    record.group_counts[index])
    axs.set_title(title_text)
    if top:
        axs.set_title("\n".join([""] * 3 + [title_text]))
    if left:
        axs.set_ylabel('Percentage of data points')
    if bot:
        axs.set_xlabel("Group size (<=)")


def plot_count_bin_sizes(axs, record: BvftRecord, resolution, bins, plot_loc=None):
    if plot_loc is None:
        plot_loc = (False, False, False, False)
    top, bot, left, right = plot_loc
    if resolution not in record.resolutions:
        print("resolution not found in bvft record!")
        return
    index = record.resolutions.index(resolution)
    axs.bar([str(i) for i in ([1] + bins)], record.count_bin_histogram[index])
    title_text = "|D| = {}, res = {:.3f}, {:.4} groups avg.".format(record.data_size, resolution,
                                                                    record.group_counts[index])
    axs.set_title(title_text)
    if top:
        axs.set_title("\n".join([""] * 3 + [title_text]))
    if left:
        axs.set_ylabel('Number of groups')
    if bot:
        axs.set_xlabel("Group size (<=)")

def plot_metric_vs_bvft_loss_plot(axs, record: BvftRecord, resolution, metric="performance", plot_loc=None):
    if plot_loc is None:
        plot_loc = (False, False, False, False)
    if resolution not in record.resolutions:
        print("resolution not found in bvft record!")
        return
    idx = record.resolutions.index(resolution)
    loss = record.losses[idx]
    loss_wo_q_star = np.max(record.loss_matrices[idx][:-1, :-1], axis=1)
    color1 = 'b'
    color2 = 'r'
    values = None
    if metric == "performance":
        values = np.max(record.model_values) - record.model_values
    elif metric == "|Q-Q*|":
        values = np.array(record.q_star_diff)
    elif metric == "|Q-TQ|":
        values = np.array(record.bellman_error)
    title_text = "|D| = {}, res = {:.3f}, {:.4} groups avg.".format(record.data_size, resolution,
                                                                    record.group_counts[idx])
    axs.set_title(title_text)
    top, bot, left, right = plot_loc
    if bot:
        axs.set_xlabel('BVFT loss')
    if left:
        axs.set_ylabel(metric)
    if top:
        axs.set_title("\n".join([""] * 3 + [title_text]))

    rank = np.argsort(loss)
    rank_wo_q_star = np.argsort(loss_wo_q_star)

    axs.plot(loss[rank], values[rank], linestyle='--', marker='o', color=color1, label=F"{metric} with Q*")
    axs.plot(loss_wo_q_star[rank_wo_q_star], values[rank_wo_q_star], linestyle='--', marker='o', color=color2, label=F"{metric} w/o Q*")
    for i, v in enumerate(values[:-1]):
        axs.plot([loss[i], loss_wo_q_star[i]], [v, v], color=color2)
    axs.legend()


def plot_metric_vs_bellman_residual_plot(axs, record: BvftRecord, resolution, metric="V(Q*) - V(Q)", plot_loc=None):
    if plot_loc is None:
        plot_loc = (False, False, False, False)
    if resolution not in record.resolutions:
        print("resolution not found in bvft record!")
        return
    idx = record.resolutions.index(resolution)
    loss = record.losses[idx]
    loss_wo_q_star = np.max(record.loss_matrices[idx][:-1, :-1], axis=1)
    color1 = 'b'
    color2 = 'r'
    values = None
    if metric == "V(Q*) - V(Q)":
        values = np.max(record.model_values) - record.model_values
    elif metric == "|Q-Q*|":
        values = record.q_star_diff
    elif metric == "|Q-TQ":
        values = record.bellman_error
    top, bot, left, right = plot_loc
    title_text = "|D| = {}, res = {:.3f}, {:.4} groups avg.".format(record.data_size, resolution, record.group_counts[idx])
    axs.set_title(title_text)
    axs.set_xlabel('Bell residual')
    if left:
        axs.set_ylabel(metric)
    if top:
        axs.set_title("\n".join([""] * 3 + [title_text]))

    rank = np.argsort(loss)
    rank_wo_q_star = np.argsort(loss_wo_q_star)
    axs.plot(loss[rank], values[rank], linestyle='--', marker='o', color=color1, label=F"{metric} with Q*")
    axs.plot(loss_wo_q_star[rank_wo_q_star], values[rank_wo_q_star], linestyle='--', marker='o', color=color2, label=F"{metric} w/o Q*")
    for i, v in enumerate(values[:-1]):
        axs.plot([loss[i], loss_wo_q_star[i]], [v, v], color=color2)
    axs.legend()


def plot_bvft_loss_vs_resolution_plot(axs, record: BvftRecord, plot_loc=None, exclude_q_star=False, model_count=5):
    if plot_loc is None:
        plot_loc = (False, False, False, False)
    losses = []
    sorted_res = sorted(record.resolutions)
    for res in sorted_res:
        i = record.resolutions.index(res)
        if exclude_q_star:
            losses.append(np.max(record.loss_matrices[i][:-1, :-1], axis=1))
        else:
            losses.append(record.losses[i])
    models = [i for i in range(record.model_count)]
    if exclude_q_star:
        models.pop()
    models_to_plot = random.sample(models, model_count)
    top, bot, left, right = plot_loc
    title_text = "|D| = {}".format(record.data_size)
    axs.set_title(title_text)
    if top:
        axs.set_title("\n".join([""] * 3 + [title_text]))
    if left:
        axs.set_ylabel("BVFT loss")
    if bot:
        axs.set_xlabel("resolution")
    axs.set_xticks(sorted_res)
    for model in models_to_plot:
        axs.plot(sorted_res, [loss[model] for loss in losses])


def plot_performance_and_q_vs_loss_scatter(axs, record: BvftRecord, resolution, q_diff, plot_loc=None, plot=True, fit=False):
    if plot_loc is None:
        plot_loc = (False, False, False, False)
    top, bot, left, right = plot_loc
    if resolution not in record.resolutions:
        print("resolution not found in bvft record!")
        return
    index = record.resolutions.index(resolution)
    ranks = np.argsort(record.losses[index])

    color1 = 'b'
    color2 = 'r'
    title_text = "|D| = {}, res = {:.3f}, {:.4} groups avg.".format(record.data_size, resolution,
                                                                    record.group_counts[index])
    axs.set_title(title_text)

    if bot:
        axs.set_xlabel('BVFT loss')
    if left:
        axs.set_ylabel('V(Q*) - V(Q)', color=color1)
    axs.tick_params(axis='y', labelcolor=color1)

    axs2 = axs.twinx()
    if right:
        axs2.set_ylabel('|q - Q*|', color=color2)
    axs2.tick_params(axis='y', labelcolor=color2)
    if top:
        axs.set_title("\n".join([""] * 3 + [title_text]))
    values = np.max(record.model_values) - record.model_values
    if plot:
        axs.plot(record.losses[index][ranks], values[ranks], linestyle='--', marker='o', color=color1)
        axs2.plot(record.losses[index][ranks], q_diff[ranks], linestyle='--', marker='o', color=color2)
    else:
        axs.scatter(record.losses[index][ranks], values[ranks], marker='o', color=color1)
        axs2.scatter(record.losses[index][ranks], q_diff[ranks], marker='o', color=color2, label='|q - Q*|')

    if fit:
        x = record.losses[index][ranks]
        y = record.model_values[ranks]
        coef = np.polyfit(x, y, 1)
        poly1d_fn = np.poly1d(coef)
        axs.plot(x, y, 'yo', x, poly1d_fn(x), '--k', color=color1)

        y = q_diff[ranks]
        coef = np.polyfit(x, y, 1)
        poly1d_fn = np.poly1d(coef)
        axs2.plot(x, y, 'yo', x, poly1d_fn(x), '--k', color=color2)


def plot_top_k_metrics(axs, records, resolutions=None, exclude_q_star=False, ks=None, plot_loc=None, auto_res=False, include_avgqsa=True):
    c = 0.1
    if plot_loc is None:
        plot_loc = (False, False, False, False)
    top, bot, left, right = plot_loc
    # metrics_names = ["Top k precision", "Top k accuracy", "Top k correlation", "Top k regret"]
    metrics_names = ["Top k precision", "Top k regret"]
    methods = [calculate_precision, calculate_regret]
    if ks is None:
        ks = [i for i in range(1, 6)]
    if resolutions is None:
        resolution_set = set()
        for record in records:
            resolution_set.update(record.resolutions)
        resolutions = sorted(list(resolution_set))
    ranker_list = ["Random", "1 sample BR", "|Q-Q*|", "|Q-TQ|", "Avg(Q(s,a))"]
    if auto_res:
        ranker_list.append("BVFT auto")
        ranker_list.append("BVFT Skyline")
        metrics_names = [n + ", bvft auto resolution" for n in metrics_names]
    else:
        ranker_list += [F"BVFT res={res}" for res in resolutions]
    ranker_metrics = OrderedDict([(name, []) for name in ranker_list])

    for record in records:
        values = record.model_values
        if exclude_q_star:
            values = values[:-1]
        ranker_loss_list = []
        ranker_loss_list.append(("Random",
                                 np.random.shuffle(np.arange(record.model_count-1)) if exclude_q_star else np.random.shuffle(np.arange(record.model_count))))
        ranker_loss_list.append(("1 sample BR",
                                 record.bellman_residual[:-1] if exclude_q_star else record.bellman_residual))
        if record.q_star_diff is not None:
            ranker_loss_list.append(("|Q-Q*|", record.q_star_diff[:-1] if exclude_q_star else record.q_star_diff))
        if record.bellman_error is not None:
            ranker_loss_list.append(("|Q-TQ|", record.bellman_error[:-1] if exclude_q_star else record.bellman_error))

        ranker_loss_list.append(("Avg(Q(s,a))", -np.array(record.avg_q[:-1]) if exclude_q_star else -np.array(record.avg_q)))
        if auto_res:
            auto_loss = []
            for i, res in enumerate(record.resolutions):
                loss = record.losses[i]
                if exclude_q_star:
                    loss = np.max(record.loss_matrices[i][:-1, :-1], axis=1)
                loss /= np.sqrt(record.data_size)
                auto_loss.append(loss + c * res)
            ranker_loss_list.append((F"BVFT auto", np.min(auto_loss, axis=0)))

            if record.optimal_grouping_skyline is not None and len(record.optimal_grouping_skyline) > 0:
                auto_loss_skyline = []
                for i, res in enumerate(record.resolutions):
                    auto_loss_skyline.append(record.optimal_grouping_skyline[i] + c * res)
                ranker_loss_list.append((F"BVFT Skyline", np.min(auto_loss_skyline, axis=0)))
        else:
            for i, res in enumerate(record.resolutions):
                loss = record.losses[i]
                if exclude_q_star:
                    loss = np.max(record.loss_matrices[i][:-1, :-1], axis=1)
                ranker_loss_list.append((F"BVFT res={res}", loss))
        for ranker, loss in ranker_loss_list:
            ranks = list(np.argsort(loss))
            metrics = calculate_top_k_metrics(methods, ranks, values, ks)
            ranker_metrics[ranker].append(metrics)
    if not include_avgqsa:
        ranker_metrics.pop("Avg(Q(s,a))")

    for i, metrics_name in enumerate(metrics_names):
        if top:
            axs[i].set_title("\n".join([""]*3 + [metrics_name]))
        if bot:
            axs[i].set_xlabel("k")
        for ranker in ranker_metrics:
            if len(ranker_metrics[ranker]) == 0:
                continue
            aggregated_metrics = np.mean(np.array(ranker_metrics[ranker]), axis=0)
            metrics_error = sem(np.array(ranker_metrics[ranker]), axis=0)
            axs[i].errorbar(ks, aggregated_metrics[i], yerr=metrics_error[i], linestyle='--', marker='o', label=ranker)
        axs[i].legend()


def calculate_top_k_metrics(metrics, ranks, values, ks):
    stats = [np.zeros(len(ks)) for i in metrics]
    for i, k in enumerate(ks):
        for j, metric in enumerate(metrics):
            stats[j][i] = metric(ranks, values, k)
    return stats


def calculate_precision(proposed_rank, values, k):
    true_rank = np.argsort(-values)
    return len(set(true_rank[:k]).intersection(proposed_rank[:k])) / float(k)


def calculate_accuracy(proposed_rank, values, k):
    true_rank = np.argsort(-values)
    return np.sum([1.0 if proposed_rank[i] == true_rank[i] else 0 for i in range(k)]) / k


def calculate_correlation(proposed_rank, values, k):
    if k == 1:
        return 1.0
    true_rank = np.argsort(-values)
    top_k_rank = [proposed_rank.index(true_rank[i]) for i in range(k)]
    return pearsonr(top_k_rank, [i for i in range(k)])[0]


def calculate_regret(ranks, values, k):
    return np.max(values) - np.max(values[ranks[:k]])


def get_subplots(row, col, title):
    res = 5
    fig, axs = plt.subplots(row, col, figsize=(col * res, row * res))
    fig.subplots_adjust(top=0.8)
    fig.suptitle(title, fontsize=20)
    return fig, axs


def get_projected_bellman_loss(Q, T, R, gamma):
    expected_R = np.sum(T * R, axis=2)
    max_v = np.max(Q, axis=1)
    TQ = expected_R + gamma * np.sum(T * max_v, axis=2)
    return np.linalg.norm(Q - TQ, 2)


if __name__ == '__main__':
    br1 = BvftRecord(data_size=100, data_explore_rate=0.5, gamma=0.95, model_count=10)
    br2 = BvftRecord(data_size=100, data_explore_rate=0.5, gamma=0.95, model_count=10)
    results = [br1, br2]
    dir_path = os.path.dirname(os.path.realpath(__file__))
    outfile = open(dir_path + "/test_pickle", "wb")
    pickle.dump(results, outfile)
    outfile.close()
    outfile = open(dir_path + "/test_pickle", "rb")
    new_results = pickle.load(outfile)
    outfile.close()
    print(new_results[1].match(new_results[0]))

    T = np.load(dir_path + "/data/taxi/ENV_T_epsilon_0.001.npy")
    R = np.load(dir_path + "/data/taxi/ENV_R_epsilon_0.001.npy")
    Q = np.load(dir_path + "/data/taxi-random/OPTIMAL_Q_GAMMA_0.99_VALUE_1.2088.npy")
    Q2 = np.load(dir_path + "/data/taxi-random/q37085_VALUE_-0.80615.npy")
    print(get_projected_bellman_loss(Q, T, R, 0.99))
    print(get_projected_bellman_loss(Q2, T, R, 0.99))

    env = gym.make('Taxi-v3')
    dir_path = os.path.dirname(os.path.realpath(__file__))
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    R = np.zeros([num_states, num_actions, num_states])
    T = np.zeros([num_states, num_actions, num_states])

    for state in range(num_states):
        for action in range(num_actions):
            for transition in env.env.P[state][action]:
                probability, next_state, reward, done = transition
                R[state, action, next_state] = reward
                T[state, action, next_state] = probability
            T[state, action, :] /= np.sum(T[state, action, :])

    optimal_q_name = "OPTIMAL_Q_GAMMA_0.95_VALUE_0.086563.npy"
    Q = np.load(dir_path + "/data/taxi-v3/" + optimal_q_name)
    Q2 = np.load(dir_path + "/data/taxi-v3/q38270_50000_VALUE_0.03675933.npy")
    print(get_projected_bellman_loss(Q, T, R, 0.95))
    print(get_projected_bellman_loss(Q2, T, R, 0.95))