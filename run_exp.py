from BVFT import BVFT
import pickle, random
import numpy as np
from keras.models import Model, load_model
from os import listdir
from os.path import isfile, join
import os
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# Define Env Variables
# ENV_NAME = 'cartpole_new'
ENV_NAME = 'lunarlander'
GAMMA = 0.99
RMAX, RMIN = 1.0, -1.0
PATH = F"data/{ENV_NAME}/"


def get_file_names(keywords):
    # get baselines and dataset
    files = [f for f in listdir(PATH) if isfile(join(PATH, f))]
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


def get_models(files, n=0):
    if n > 0:
        random.shuffle(files)
    models = []
    values = []
    for i, f in enumerate(files):
        models.append(load_model(PATH + f))
        values.append(float(f.split("_")[-1][:-3]))
        if 0 < n == i+1:
            break
    return models, np.array(values)


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
    q_functions, values = get_models(model_names, n=num_models)
    data_names = get_file_names(data_keywords)
    res_size = len(resolutions)
    bins = [2, 3, 4, 8, 16, 100, 1e5]

    fig_rank, axs_rank = plt.subplots(res_size + 1, len(data_sizes))
    fig_rank.set_size_inches(20, 12)
    fig_rank.suptitle("BVFT ranking vs actual rollout estimate", fontsize=20)
    for ax in axs_rank.flat:
        ax.set(ylabel='Actual value advantage')

    fig_bin, axs_bin = plt.subplots(res_size, len(data_sizes))
    fig_bin.set_size_inches(20, 12)
    fig_bin.suptitle("BVFT group size distributions", fontsize=20)
    for ax in axs_bin.flat:
        ax.set(ylabel='Percentage of data points', xlabel="Group size (<=)")

    dataset = get_data(data_names, size=max(data_sizes))
    for j, data_size in enumerate(data_sizes):
        data = random.sample(dataset, data_size)
        bvft = BVFT(q_functions, data, GAMMA, RMAX, RMIN, bins=bins)

        for i, res in enumerate(resolutions):
            ranks, loss_matrix, bin_histo = bvft.run(resolution=res)
            print(np.sort(np.max(loss_matrix, axis=1)))
            axs_rank[i, j].bar([i+1 for i in range(num_models)], values[ranks] - np.mean(values))
            axs_rank[i, j].set_title("samples = {}, resolution = {:.5f}".format(data_size, res))
            axs_bin[i, j].bar([str(i) for i in ([1] + bins)], bin_histo)
            axs_bin[i, j].set_title("samples = {}, resolution = {:.5f}".format(data_size, res))

        br_rank = bvft.get_br_ranking()
        axs_rank[res_size, j].bar([i + 1 for i in range(num_models)], values[br_rank] - np.mean(values))
        axs_rank[res_size, j].set_title(F"data = {data_size} samples, Bellman residual")
        axs_rank[res_size, j].set(xlabel='Ranking')
    fig_rank.show()
    fig_bin.show()


model_keywords = ["lunarlander_DQN", ".h5", "VALUE"]
data_keywords = ["DATA"]
data_sizes = [10**n for n in range(4, 6)] + [2*10**5]

resolutions = [1e-1**n for n in range(1, 4)]
resolutions = [2.0, 0.5, 0.1, 0.05]

model_counts = [5 for j in range(15)]

for num_models in model_counts:
    experiment1(model_keywords, data_keywords, num_models, data_sizes, resolutions)