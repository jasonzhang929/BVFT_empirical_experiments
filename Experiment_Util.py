from BVFT import BVFT
import pickle, random, time
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from BvftUtil import *
from keras.models import Model, load_model
import tensorflow as tf


class BVFTExperiment(object):
    def __init__(self, folder_name):
        self.folder_name = folder_name
        self.PATH = F"data/{folder_name}/"
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.TOP_Q_FLOOR = None
        self.NORMAL_Q_CEILING = None
        self.NORMAL_Q_FLOOR = None
        self.model_keywords = None
        self.data_keywords = None
        self.data_sizes = None
        self.GAMMA = None
        self.RMAX = None
        self.RMIN = None
        self.bins = [2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1e5]


    def get_file_names(self, keywords, path=None):
        # get baselines and dataset
        if path is None:
            path = self.PATH
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

    def get_records(self, files, folder=""):
        records = []
        for f in files:
            file = open(self.dir_path + "/data/bvft/" + folder + f, "rb")
            records += pickle.load(file)
            file.close()
        return records

    def get_all_model_values(self, files):
        return [float(f.split("_")[-1][:-3]) for f in files]

    def get_models(self, files, n=10, top_q_count=2, model_gap=20.0):
        random.shuffle(files)
        model_values = [float(f.split("_")[-1][:-3]) for f in files]
        selected_model_names = []
        selected_model_values = []
        selected_model_functions = []

        for i, v in enumerate(model_values):
            if v >= self.TOP_Q_FLOOR:
                selected_model_names.append(files[i])
                selected_model_values.append(v)
                selected_model_functions.append(load_model(self.PATH + files[i]))
                if len(selected_model_names) == top_q_count:
                    break
        for i, v in enumerate(model_values):
            if self.NORMAL_Q_FLOOR < v <= self.NORMAL_Q_CEILING:
                skip = False
                for v1 in selected_model_values:
                    if abs(v1 - v) < model_gap:
                        skip = True
                        break
                if skip:
                    continue
                selected_model_names.append(files[i])
                selected_model_values.append(v)
                selected_model_functions.append(load_model(self.PATH + files[i]))
                if len(selected_model_names) == n:
                    break
        if len(selected_model_names) < n:
            print("NOT ENOUGH MODEL!")
        return selected_model_functions, selected_model_values, selected_model_names

    def get_data(self, files, size=0):
        data = []
        if size > 0:
            random.shuffle(files)
        for file in files:
            outfile = open(self.PATH + file, "rb")
            data += pickle.load(outfile)
            outfile.close()
            if 0 < size < len(data):
                break
        return data

    def experiment2(self, num_model, data_size, num_runs, data_explore_rate, resolutions):
        model_names = self.get_file_names(self.model_keywords)
        records = []
        k = np.random.randint(1e6)
        t = time.time()
        for run in range(num_runs):
            q_functions, values, q_names = self.get_models(model_names, n=num_model)
            data_names = self.get_file_names(["DATA", str(data_explore_rate)])
            dataset = self.get_data(data_names, size=max(self.data_sizes))
            data_start = np.random.randint(len(dataset) - data_size)
            data = dataset[data_start:data_start + data_size]
            record = BvftRecord(data_size=data_size, gamma=self.GAMMA, data_explore_rate=data_explore_rate,
                                model_count=num_model)
            record.model_values = values
            # record.q_star_diff = q_star_diff
            record.q_names = q_names
            bvft = BVFT(q_functions, data, self.GAMMA, self.RMAX, self.RMIN, record, bins=self.bins, tabular=False)

            for i, res in enumerate(resolutions):
                bvft.run(resolution=res)
            bvft.get_br_ranking()
            records.append(record)
            if run > 0 and run % 10 == 0:
                outfile = open(self.dir_path + F'/data/bvft/{self.folder_name}_records_{k}', "wb")
                pickle.dump(records, outfile)
                outfile.close()
                print(F'run {run}, saved {self.folder_name}_records_{k}, {(time.time() - t) / 60} minutes')
                t = time.time()
        outfile = open(self.dir_path + F'/data/bvft/{self.folder_name}_records_{k}', "wb")
        pickle.dump(records, outfile)
        outfile.close()