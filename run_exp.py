from BVFT import BVFT
import pickle, random
import numpy as np
from keras.models import Model, load_model
from os import listdir
from os.path import isfile, join
import os
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Define Env Variables
ENV_NAME = 'cartpole'
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
    fig_rank, axs_rank = plt.subplots(len(resolutions), len(data_sizes))
    fig_rank.set_size_inches(18.5, 10.5)
    for ax in axs_rank.flat:
        ax.set(xlabel='Rank', ylabel='Actual value advantage')

    dataset = get_data(data_names, size=max(data_sizes))
    for j, data_size in enumerate(data_sizes):
        data = random.sample(dataset, data_size)
        print(F"Data size {data_size} ========================================================")
        bvft = BVFT(q_functions, data, GAMMA, RMAX, RMIN)

        for i, res in enumerate(resolutions):
            print(F"Resolution = {res}")
            ranks, loss_matrix = bvft.run(resolution=res)
            axs_rank[i, j].bar([i+1 for i in range(num_models)], values[ranks] - np.mean(values))
            axs_rank[i, j].set_title("data = {} samples, resolution = {:.6f}".format(data_size, res))
    fig_rank.show()


model_keywords = ["cartpole_DQN", ".h5"]
data_keywords = ["data_cartpole_DQN"]
num_models = 2
data_sizes = [10**n for n in range(2, 3)]
resolutions = [1e-1**n for n in range(2, 4)]

for m in [2]:
    experiment1(model_keywords, data_keywords, m, data_sizes, resolutions)