from BVFT import BVFT
import pickle, random
import numpy as np
from keras.models import Model, load_model
from os import listdir
from os.path import isfile, join
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Define Env Variables
ENV_NAME = 'cartpole'
gamma = 0.99
rmax, rmin = 1.0, -1.0
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
        if 0 < n == i:
            break
    return models, values


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

for i in range(1):
    model_names = get_file_names([".h5"])
    q_functions, values = get_models(model_names, n=2)
    data_names = get_file_names(["data_cartpole_DQN"])
    data = get_data(data_names, size=100000)
    b = BVFT(q_functions, data, gamma, rmax, rmin)
    for res in [0.0001, 0.001]:
        print(res)
        ranks = b.run(resolution=res)[0]
        vals = [values[i] for i in ranks]
        print(vals)
