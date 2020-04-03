import os
import os.path
import dill
import torch.utils.data as data
import torch
from functools import partial

from utils.utils import mkdir_in_path, read_json, filter_keys_in_strings, list_files_abs_path, get_filename

import numpy as np

from ..db_stats import buildKeyOrder
from .audio_pairs_base_loader import AudioPairsLoader
from tqdm import tqdm, trange

import logging
from random import shuffle

from copy import deepcopy
# from audiolazy.lazy_midi import midi2str

import ipdb


FORMATS = ["wav", "mp3"]


def preprocess(preprocessing, data):
    return preprocessing(data)

class MP3ToWAV(AudioPairsLoader):
    def __init__(self, **kargs):
        AudioPairsLoader.__init__(self, **kargs)

    def __hash__(self):
        return self.preprocessing.__hash__()

    def __getitem__(self, index):
        if self.getitem_processing:
            return self.getitem_processing(self.data[index]), \
                   self.getitem_processing(self.metadata[index])
        else:
            return self.data[index], self.metadata[index]

    def get_pair(self, file):
        filename = get_filename(file)
        pair = os.path.join(self.data_path2, filename + '.wav')
        if os.path.exists(pair):
            return pair
        else:
            return None

    def get_random_labels(self, size):
        ipdb.set_trace()
        return None

    def index_to_labels(self, **args):
        return None

    def preprocess_data(self):
        import multiprocessing
        p = multiprocessing.Pool(multiprocessing.cpu_count())

        print("Preprocessing data pairs...")
        self.data = list(p.map(self.preprocessing,
                        tqdm(self.data, desc='preprocessing-loop')))

        preprocess_ = partial(preprocess, self.preprocessing)
        self.metadata = list(p.map(preprocess_,
                            tqdm(self.metadata, desc='preprocessing-loop')))
        print("Data preprocessing done")

    def load_data(self):
        files = list_files_abs_path(self.data_path, 'wav')
        data = []
        metadata = []
        for file in files:
            mp3_file = self.get_pair(file)
            if mp3_file is None:
                print(f"Pair not found for file {file}. Skipping...")
                continue
            data.append(file)
            metadata.append(mp3_file)
            if len(data) >= self.criteria['size']: break
        return data, metadata, {}
