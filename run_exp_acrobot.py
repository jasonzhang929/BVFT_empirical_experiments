from BVFT import BVFT
import pickle, random, time
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from BvftUtil import *
from keras.models import Model, load_model


def get_file_names(keywords, path=None):
    # get baselines and dataset
    if path is None:
        path = PATH
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


def get_models(files, n=0, must_include=""):
    if n > 0:
        random.shuffle(files)
    models = []
    values = []
    names = []
    for i, f in enumerate(files):
        v = float(f.split("_")[-1][:-3])
        if model_low < v < model_up:
            q = load_model(PATH + f)
            models.append(q)
            values.append(v)
            names.append(f)
        if 0 < n == len(models):
            break
    if must_include != "":
        models[-1] = np.load(PATH + must_include)
        values[-1] = float(must_include.split("_")[-1][:-4])
        names[-1] = must_include
    return models, np.array(values), names


def get_T_R():
    T = np.load(dir_path + F"/data/{ENV_NAME}/ENV_T_epsilon_0.001.npy")
    R = np.load(dir_path + F"/data/{ENV_NAME}/ENV_R_epsilon_0.001.npy")
    return T, R


# def get_data(files, size=0):
#     if size > 0:
#         random.shuffle(files)
#     data = np.load(PATH + files[0])
#     for file in files[1:]:
#         data = np.concatenate((data, np.load(PATH + file)), axis=0)
#         if 0 < size < len(data):
#             break
#     return data


def get_data(files, size=0):
    data = []
    if size > 0:
        random.shuffle(files)
    for file in files:
        outfile = open(PATH + file, "rb")
        data += pickle.load(outfile)
        outfile.close()
        if 0 < size < len(data):
            break
    return data

def experiment1(model_keywords, data_keywords, num_models, data_sizes, resolutions):
    model_names = get_file_names(model_keywords)
    q_functions, values, _ = get_models(model_names, n=num_models, must_include=optimal_q_name)

    optimal_q = q_functions[-1]
    q_star_diff = np.array([np.linalg.norm(optimal_q - q, 2) for q in q_functions])
    data_names = get_file_names(data_keywords)
    res_size = len(resolutions)

    fig_rank, axs_rank = get_subplots(res_size + 1, len(data_sizes), "BVFT ranking vs actual rollout estimate")
    fig_bin, axs_bin = get_subplots(res_size, len(data_sizes), "BVFT group size distributions")

    fig_per, axs_per = get_subplots(res_size, len(data_sizes), "Q policy value, Q* distance vs BVVF loss")

    dataset = get_data(data_names, size=max(data_sizes))
    for j, data_size in enumerate(data_sizes):
        data_start = np.random.randint(len(dataset) - data_size)
        data = dataset[data_start:data_start + data_size]
        record = BvftRecord(data_size=data_size, gamma=GAMMA, data_explore_rate=data_explore_rate,
                            model_count=num_models)
        record.model_values = values
        bvft = BVFT(q_functions, data, GAMMA, RMAX, RMIN, record, bins=bins, tabular=True)

        for i, res in enumerate(resolutions):
            bvft.run(resolution=res)
            left = j == 0
            right = j == len(data_sizes) - 1
            bot = i == len(resolutions) - 1
            plot_performance_vs_rank_bar(axs_rank[i, j], record, res)
            plot_count_bin_sizes(axs_bin[i, j], record, res, bins)
            plot_performance_and_q_vs_loss_scatter(axs_per[i, j], record, res, q_star_diff, left=left, right=right,
                                                   bot=bot)

        br_rank = bvft.get_br_ranking()
        print(br_rank)
        axs_rank[res_size, j].bar([i + 1 for i in range(num_models)], values[br_rank] - np.mean(values))
        axs_rank[res_size, j].set_title(F"data = {data_size} samples, Bellman residual")
        axs_rank[res_size, j].set(xlabel='Ranking')

    fig_bin.show()
    fig_per.show()


def generate_more_q(count=1):
    model_names = get_file_names(model_keywords)
    optimal_q = np.load(PATH + optimal_q_name)
    q_functions, values, q_names = get_models(model_names, n=len(model_names))
    row, col = np.shape(optimal_q)
    for i, q in enumerate(q_functions):
        for j in range(count):
            diff = optimal_q - q
            r = random.random() * 0.5 + 0.5
            new_q = (r + np.random.rand(row, col) * 0.05) * diff + q
            name = "SYN_{:.5}_{}.npy".format(r, values[i])
            np.save(PATH + name, new_q)


def experiment2(num_model, data_size, num_runs, data_explore_rate, resolutions):
    model_names = get_file_names(model_keywords)
    records = []
    k = np.random.randint(1e6)
    t = time.time()
    for run in range(num_runs):
        q_functions, values, q_names = get_models(model_names, n=num_model, must_include=optimal_q_name)
        # optimal_q = q_functions[-1]
        # q_star_diff = np.array([np.linalg.norm(optimal_q - q, 2) for q in q_functions])
        data_keywords = ["DATA", "EPS_" + str(data_explore_rate)]
        # data_keywords = ["DATA"]
        data_names = get_file_names(data_keywords)
        dataset = get_data(data_names, size=max(data_sizes))
        data_start = np.random.randint(len(dataset) - data_size)
        data = dataset[data_start:data_start + data_size]
        record = BvftRecord(data_size=data_size, gamma=GAMMA, data_explore_rate=data_explore_rate,
                            model_count=num_model)
        record.model_values = values
        # record.q_star_diff = q_star_diff
        record.q_names = q_names
        bvft = BVFT(q_functions, data, GAMMA, RMAX, RMIN, record, bins=bins, tabular=False)

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
    num_models = [10, 15]
    data_sizes = [500, 5000, 50000]

    data_explore_rates = [0.5, 1.0]
    resolutions = {5000: [0.1, 0.2, 0.5, 0.7, 1.0, 3.0],
                   500: [0.1, 0.2, 0.5, 0.7, 1.0, 3.0],
                   50000: [0.1, 0.2, 0.5, 0.7, 1.0, 3.0]}

    for num_model in num_models:
        for data_explore_rate in data_explore_rates:
            for data_size in data_sizes:
                print(F"num_model {num_model}, data_explore_rate {data_explore_rate}, data_size {data_size}")
                experiment2(num_model, data_size, num_runs, data_explore_rate, resolutions[data_size])


def experiment3(model_count, folder="", auto_res=False):
    record_files = get_file_names([ENV_NAME], path="data/bvft/" + folder)
    records = get_records(record_files, folder=folder)
    data_sizes = [500, 5000, 50000]
    data_explore_rates = [0.2, 0.5, 1.0]
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
                               auto_res=False)
            if auto_res:
                plot_top_k_metrics(axs[i][2:], matched_records, exclude_q_star=not include_q_star, plot_loc=plot_loc,
                                   auto_res=auto_res)
            axs[i, 0].set_ylabel(F"|D| = {data_size}")
            for j, record in enumerate(random.sample(matched_records, k)):
                top, bot, left, right = j == 0, j == k - 1, i == 0, i == len(data_sizes) - 1
                plot_loc = (top, bot, left, right)
                plot_bvft_loss_vs_resolution_plot(axs_res[j, i], record, plot_loc=plot_loc,
                                                  exclude_q_star=not include_q_star, model_count=8)
        plt.show()
    model_values = list(model_stats.values())
    plt.hist(model_values, 50, density=True, facecolor='g', alpha=0.75)
    plt.xlabel('Roll out values')
    plt.ylabel('Model count')
    plt.title('Histogram of model values')
    plt.grid(True)
    plt.show()


def fill_bellman_error():
    mem = {}
    record_files = get_file_names([ENV_NAME], path="data/bvft/")
    T = np.load(dir_path + F"/data/{ENV_NAME}/ENV_T_epsilon_0.001.npy")
    R = np.load(dir_path + F"/data/{ENV_NAME}/ENV_R_epsilon_0.001.npy")
    for file in record_files:
        file_path = open(dir_path + "/data/bvft/" + file, "rb")
        records = pickle.load(file_path)
        file_path.close()
        print(len(records))
        q_set = set()
        for i, record in enumerate(records):
            q_set.update(record.q_names)
            if record.bellman_error is not None:
                continue
            bellman_errors = []
            record.q_names[-1] = optimal_q_name
            for q_name in record.q_names:
                if q_name not in mem:
                    q = np.load(dir_path + F"/data/{ENV_NAME}/" + q_name)
                    mem[q_name] = get_projected_bellman_loss(q, T, R, GAMMA)
                bellman_errors.append(mem[q_name])
            record.bellman_error = bellman_errors
        file_path = open(dir_path + "/data/bvft/" + file, "wb")
        pickle.dump(records, file_path)
        file_path.close()
        print(file, len(q_set))


def experiment4(num_model):
    model_names = get_file_names(model_keywords)
    q_functions, values, q_names = get_models(model_names, n=num_model, must_include=optimal_q_name)
    # optimal_q = q_functions[-1]
    # q_star_diff = np.array([np.linalg.norm(optimal_q - q, 2) for q in q_functions])
    # T, R = get_T_R()
    # bellman_error = [get_projected_bellman_loss(q, T, R, GAMMA) for q in q_functions]

    data_names = get_file_names(data_keywords)
    res_size = len(resolutions)

    title_prefix = F"{ENV_NAME} data explore rate {data_explore_rate}, "
    fig_perf_bvft, axs_perf_bvft = get_subplots(res_size, len(data_sizes), title_prefix + "V(Q*) - V(Q) vs BVFT loss")
    # fig_q_star_diff_bvft, axs_q_star_diff_bvft = get_subplots(res_size, len(data_sizes),
    #                                                           title_prefix + "|Q - Q*| vs BVFT loss")
    # fig_berror_vs_bvft, axs_berror_vs_bvft = get_subplots(res_size, len(data_sizes),
    #                                                       title_prefix + "|Q - TQ| vs BVFT loss")
    fig_bin_percent, axs_bin_percent = get_subplots(res_size, len(data_sizes),
                                                    title_prefix + "BVFT group size distributions")
    fig_bin_count, axs_bin_count = get_subplots(res_size, len(data_sizes),
                                                title_prefix + "BVFT group size distributions")
    # fig_per, axs_per = get_subplots(res_size, len(data_sizes), title_prefix + "V(Q*) - V(Q), |Q - Q*| vs BVFT loss")
    include_q_star = False
    text = "Q* included" if include_q_star else "Q* excluded"
    fig_res, axs_res = get_subplots(1, len(data_sizes),
                                    F"{ENV_NAME}, {num_model} models, data exploration rate {data_explore_rate}, {text}")

    dataset = get_data(data_names, size=max(data_sizes))
    for j, data_size in enumerate(data_sizes):
        data_start = np.random.randint(len(dataset) - data_size)
        data = dataset[data_start:data_start + data_size]
        record = BvftRecord(data_size=data_size, gamma=GAMMA, data_explore_rate=data_explore_rate,
                            model_count=num_model)
        record.model_values = values
        # record.q_star_diff = q_star_diff
        # record.bellman_error = bellman_error
        bvft = BVFT(q_functions, data, GAMMA, RMAX, RMIN, record, bins=bins, tabular=False, verbose=True)

        for i, res in enumerate(resolutions):
            bvft.run(resolution=res)
            bvft.get_br_ranking()
            top, bot, left, right = i == 0, i == len(resolutions) - 1, j == 0, j == len(data_sizes) - 1
            plot_loc = (top, bot, left, right)
            plot_metric_vs_bvft_loss_plot(axs_perf_bvft[i, j], record, res, plot_loc=plot_loc)
            # plot_metric_vs_bvft_loss_plot(axs_q_star_diff_bvft[i, j], record, res, plot_loc=plot_loc, metric="|Q-Q*|")
            # plot_metric_vs_bvft_loss_plot(axs_berror_vs_bvft[i, j], record, res, plot_loc=plot_loc, metric="|Q-TQ|")
            plot_percent_bin_sizes(axs_bin_percent[i, j], record, res, bins, plot_loc=plot_loc)
            plot_count_bin_sizes(axs_bin_count[i, j], record, res, bins, plot_loc=plot_loc)
            # plot_performance_and_q_vs_loss_scatter(axs_per[i, j], record, res, q_star_diff, plot_loc=plot_loc)
        plot_loc = (True, True, j == 0, j == len(data_sizes) - 1)
        mc = num_model if include_q_star else num_model - 1
        plot_bvft_loss_vs_resolution_plot(axs_res[j], record, exclude_q_star=not include_q_star, model_count=mc,
                                          plot_loc=plot_loc)

    fig_perf_bvft.show()
    # fig_q_star_diff_bvft.show()
    # fig_berror_vs_bvft.show()
    fig_bin_count.show()
    fig_bin_percent.show()
    # fig_per.show()
    fig_res.show()


if __name__ == '__main__':
    # ENV_NAME = 'taxi-random-0.5'
    # optimal_q_name = "OPTIMAL_Q_GAMMA_0.99_VALUE_-0.43706.npy"

    ENV_NAME = 'acrobot'
    optimal_q_name = ""


    PATH = F"data/{ENV_NAME}/"
    GAMMA = 0.99
    RMAX, RMIN = 0.0, -1.0
    dir_path = os.path.dirname(os.path.realpath(__file__))

    include_q_star = True

    bins = [2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1e5]
    data_explore_rate = 1.0
    model_keywords = ["DQN", "VALUE", ".h5"]
    data_keywords = ["DATA"]
    data_sizes = [500, 5000, 50000]

    resolutions = np.array([0.1, 0.2, 0.5, 0.7, 1.0, 3.0])

    model_counts = [10, 15]
    model_up = 1.0
    model_low = -200

    # data_names = get_file_names(data_keywords)
    # dataset = get_data(data_names, size=100000)
    # counts = [0,0]
    #
    # for d in dataset:
    #     if d[2] == -1:
    #         counts[0] += 1
    #     if d[2] == 20:
    #         counts[1] += 1
    # print(counts)

    # for num_models in model_counts:
    #     experiment1(model_keywords, data_keywords, num_models, data_sizes, resolutions)
    # tm = time.time()
    # for i in range(10):
    #     print(F"I {i} {(time.time() - tm)/3600}")
    #     run_experiment_2(30)

    # fill_bellman_error()
    experiment3(15, auto_res=True, folder="")
    # for num_model in model_counts:
    #     experiment4(num_model)
    # generate_more_q(count=1)