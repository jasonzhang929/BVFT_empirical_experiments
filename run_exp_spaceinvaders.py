from BVFT import BVFT
import pickle, random, time
import numpy as np
from os import listdir
from os.path import isfile, join
from BvftUtil import *
from baseline_scripts.DQN import Conv_Q
from baseline_scripts.BCQ_utils import ReplayBuffer
import torch

def get_file_names(keywords, path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    results = []

    for file in files:
        match = True
        for keyword in keywords:
            if keyword not in file:
                match = False
                break
        if match:
            results.append(file)
    if len(results) == 0:
        print(F'Found 0 files with keywords: {" ".join(keywords)}')
    return results


def get_records(files, folder=""):
    records = []
    for f in files:
        file = open(dir_path + "/data/bvft/" + folder + f, "rb")
        records += pickle.load(file)
        file.close()
    return records


def get_all_model_values(files):
    return [float(f.split("_")[-2]) for f in files]


def get_models(files, n=10, top_q_count=2):
    random.shuffle(files)
    model_values = [float(f.split("_")[-2]) for f in files]
    selected_model_names = []
    selected_model_values = []
    selected_model_functions = []

    for i, v in enumerate(model_values):
        if v >= TOP_Q_FLOOR:
            selected_model_names.append(files[i])
            selected_model_values.append(v)
            model = Conv_Q(4, action_space).to(device)
            model.load_state_dict(torch.load(model_path + files[i]))
            selected_model_functions.append(model)
            if len(selected_model_names) == top_q_count:
                break
    for i, v in enumerate(model_values):
        if NORMAL_Q_FLOOR < v <= NORMAL_Q_CEILING:
            skip = False
            for v1 in selected_model_values:
                if abs(v1-v) < model_gap:
                    skip = True
                    break
            if skip:
                continue
            selected_model_names.append(files[i])
            selected_model_values.append(v)
            model = Conv_Q(4, action_space).to(device)
            model.load_state_dict(torch.load(model_path + files[i]))
            selected_model_functions.append(model)
            if len(selected_model_names) == n:
                break
    if len(selected_model_names) < n:
        print("NOT ENOUGH MODEL!")
    return selected_model_functions, selected_model_values, selected_model_names


def get_data(files, size=0):
    replay_buffer = ReplayBuffer(state_dim, True, atari_preprocessing, 128,
                                           1e6, device)
    if size > 0:
        random.shuffle(files)
    buffer_name = "_".join(files[0].split("_")[:-1])
    replay_buffer.load(f"./buffers/{buffer_name}")
    return replay_buffer


def experiment2(num_model, data_size, num_runs, data_explore_rate, resolutions):
    model_names = get_file_names(model_keywords, model_path)
    records = []
    k = np.random.randint(1e6)
    t = time.time()
    data_names = get_file_names([ENV_NAME, 'action', str(data_explore_rate), '500000'], data_path)
    dataset = get_data(data_names, size=max(data_sizes))
    for run in range(num_runs):
        q_functions, values, q_names = get_models(model_names, n=num_model)
        record = BvftRecord(data_size=data_size, gamma=GAMMA, data_explore_rate=data_explore_rate,
                            model_count=num_model)
        record.model_values = values
        # record.q_star_diff = q_star_diff
        record.q_names = q_names
        bvft = BVFT(q_functions, dataset, GAMMA, RMAX, RMIN, record, bins=bins, q_type='torch_atari', data_size=data_size)

        for i, res in enumerate(resolutions):
            bvft.run(resolution=res)
        bvft.get_br_ranking()
        records.append(record)
        if run > 0 and run % 10 == 0:
            outfile = open(dir_path + F'/data/bvft/{ENV_NAME}_records_{k}', "wb")
            pickle.dump(records, outfile)
            outfile.close()
            print(F'run {run}, saved {ENV_NAME}_records_{k}, {(time.time() - t) / 60} minutes')
            t = time.time()
    outfile = open(dir_path + F'/data/bvft/{ENV_NAME}_records_{k}', "wb")
    pickle.dump(records, outfile)
    outfile.close()


def run_experiment_2(num_runs):
    num_models = [10]
    data_sizes = [500, 50000]

    data_explore_rates = [0.0, 0.2, 0.5, 0.8, 1.0]
    resolutions = {10000: [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
                   500: [0.05, 0.1, 0.2, 0.3, 0.5],
                   50000: [0.05, 0.1, 0.2, 0.3, 0.5]}

    for num_model in num_models:
        for data_explore_rate in data_explore_rates:
            for data_size in data_sizes:
                print(F"num_model {num_model}, data_explore_rate {data_explore_rate}, data_size {data_size}")
                experiment2(num_model, data_size, num_runs, data_explore_rate, resolutions[data_size])


def experiment3(model_count, folder="", auto_res=False, c=0.1):
    record_files = get_file_names([ENV_NAME], path="data/bvft/" + folder)
    records = get_records(record_files, folder=folder)
    data_sizes = [500, 50000]
    data_explore_rates = [0.0, 0.2, 0.5, 0.8, 1.0]
    k = 4
    model_stats = {}
    for data_explore_rate in data_explore_rates:
        fig, axs = get_subplots(len(data_sizes), 4 if auto_res else 2,
                                F"{ENV_NAME}, {model_count} models, data exploration rate {data_explore_rate}")
        fig_res, axs_res = get_subplots(k, len(data_sizes),
                                        F"{ENV_NAME}, {model_count} models, data exploration rate {data_explore_rate}")
        for i, data_size in enumerate(data_sizes):
            record_matcher = BvftRecord(data_size=data_size, data_explore_rate=data_explore_rate, gamma=GAMMA,
                                        model_count=model_count)
            matched_records = []
            for record in records:
                if record_matcher.match(record):
                    matched_records.append(record)
                    for j, name in enumerate(record.q_names):
                        model_stats[name] = record.model_values[j]
            top, bot, left, right = i == 0, i == len(data_sizes) - 1, False, False
            plot_loc = (top, bot, left, right)
            plot_top_k_metrics(axs[i][:2], matched_records, exclude_q_star=not include_q_star, plot_loc=plot_loc,
                               auto_res=False, c=c)
            if auto_res:
                plot_top_k_metrics(axs[i][2:], matched_records, exclude_q_star=not include_q_star, plot_loc=plot_loc,
                                   auto_res=auto_res, c=c)
            axs[i, 0].set_ylabel(F"|D| = {data_size}")
            # for j, record in enumerate(random.sample(matched_records, k)):
            #     top, bot, left, right = j == 0, j == k - 1, i == 0, i == len(data_sizes) - 1
            #     plot_loc = (top, bot, left, right)
            #     plot_bvft_loss_vs_resolution_plot(axs_res[j, i], record, plot_loc=plot_loc,
            #                                       exclude_q_star=not include_q_star, model_count=8)
        plt.show()
    model_values = list(model_stats.values())
    plt.hist(model_values, 50, density=False, facecolor='g', alpha=0.75)
    plt.xlabel('Roll out values')
    plt.ylabel('Model count')
    plt.title('Histogram of model values')
    plt.grid(True)
    plt.show()


def experiment4(num_model):
    model_names = get_file_names(model_keywords, model_path)
    q_functions, values, q_names = get_models(model_names, n=num_model)

    data_names = get_file_names(data_keywords, data_path)
    res_size = len(resolutions)

    title_prefix = F"{ENV_NAME} data explore rate {explore_rate}, "
    fig_perf_bvft, axs_perf_bvft = get_subplots(res_size, len(data_sizes), title_prefix + "V(Q*) - V(Q) vs BVFT loss")
    fig_bin_percent, axs_bin_percent = get_subplots(res_size, len(data_sizes),
                                                    title_prefix + "BVFT group size distributions")
    fig_bin_count, axs_bin_count = get_subplots(res_size, len(data_sizes),
                                                title_prefix + "BVFT group size distributions")
    include_q_star = False
    text = "Q* included" if include_q_star else "Q* excluded"
    fig_res, axs_res = get_subplots(1, len(data_sizes),
                                    F"{ENV_NAME}, {num_model} models, data exploration rate {explore_rate}, {text}")

    dataset = get_data(data_names, size=max(data_sizes))
    for j, data_size in enumerate(data_sizes):
        record = BvftRecord(data_size=data_size, gamma=GAMMA, data_explore_rate=explore_rate,
                            model_count=num_model)
        record.model_values = values
        record.q_names = q_names
        bvft = BVFT(q_functions, dataset, GAMMA, RMAX, RMIN, record, bins=bins, q_type='torch_atari', verbose=True, data_size=data_size)

        for i, res in enumerate(resolutions):
            bvft.run(resolution=res)
            bvft.get_br_ranking()
            top, bot, left, right = i == 0, i == len(resolutions) - 1, j == 0, j == len(data_sizes) - 1
            plot_loc = (top, bot, left, right)
            plot_metric_vs_bvft_loss_plot(axs_perf_bvft[i, j], record, res, plot_loc=plot_loc)
            # plot_percent_bin_sizes(axs_bin_percent[i, j], record, res, bins, plot_loc=plot_loc)
            # plot_count_bin_sizes(axs_bin_count[i, j], record, res, bins, plot_loc=plot_loc)
        plot_loc = (True, True, j == 0, j == len(data_sizes) - 1)
        mc = num_model if include_q_star else num_model - 1
        plot_bvft_loss_vs_resolution_plot(axs_res[j], record, exclude_q_star=not include_q_star, model_count=mc,
                                          plot_loc=plot_loc, show_model_name=True)

    fig_perf_bvft.show()
    # fig_bin_count.show()
    # fig_bin_percent.show()
    fig_res.show()


def show_model_distribution():
    model_names = get_file_names(model_keywords, model_path)
    values = get_all_model_values(model_names)
    plt.hist(values, 50, density=False, facecolor='g', alpha=0.75)
    plt.xlabel('Roll out values')
    plt.ylabel('Model count')
    plt.title(F'Histogram of {ENV_NAME} model values ({len(values)} total)')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    atari_preprocessing = {
        "frame_skip": 4,
        "frame_size": 84,
        "state_history": 4,
        "done_on_life_loss": False,
        "reward_clipping": True,
        "max_episode_timesteps": 27e3
    }

    ENV_NAME = 'SpaceInvadersNoFrameskip-v0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(ENV_NAME)
    action_space = env.action_space.n
    state_dim = 4

    model_path = "./models/"
    data_path = "./buffers/"
    GAMMA = 0.99
    RMAX, RMIN = 10.0, -10.0
    dir_path = os.path.dirname(os.path.realpath(__file__))

    include_q_star = True

    bins = [2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1e5]
    explore_rate = 0.8
    model_keywords = ["DQN", ENV_NAME, "Q"]
    data_keywords = [ENV_NAME, "action", str(explore_rate)]
    data_sizes = [1000, 50000]
    # data_sizes = [20]

    resolutions = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0])

    model_counts = [10]*2
    TOP_Q_FLOOR = 570.0
    NORMAL_Q_CEILING = 530.0
    NORMAL_Q_FLOOR = 300
    model_gap = 15


    # for num_models in model_counts:
    #     experiment1(model_keywords, data_keywords, num_models, data_sizes, resolutions)
    tm = time.time()
    for i in range(10):
        print(F"I {i} {(time.time() - tm)/3600}")
        # run_experiment_2(10)
    # show_model_distribution()
    experiment3(10, auto_res=True, folder="", c=0.000)
    # for num_model in model_counts:
        # experiment4(num_model)
    # generate_more_q(count=1)