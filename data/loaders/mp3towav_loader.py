import os
import os.path
import dill
import torch.utils.data as data
import torch
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

class MP3ToWAV(AudioPairsLoader):
    def __init__(self, **kargs):
        AudioPairsLoader.__init__(self, **kargs)

    def __getitem__(self, index):
        if self.getitem_processing:
            return self.getitem_processing(self.data[index]), \
                   self.getitem_processing(self.metadata[index])
        else:
            return self.data[index], self.metadata[index]

    def get_pair(self, file):
        filename = get_filename(file)
        pair = os.path.join(self.data_path2, filename + '.mp3')
        if os.path.exists(pair):
            return pair
        else:
            return None

    def preprocess_data(self):
        print("Preprocessing data pairs...")
        self.data = list(map(self.preprocessing, 
                        tqdm(self.data, desc='preprocessing-loop')))
        
        self.metadata = list(map(self.preprocessing, 
                        tqdm(self.metadata, desc='preprocessing-loop')))
        print("Data preprocessing done")

    def read_data(self):
        files = list_files_abs_path(self.data_path, self.format)
        for file in files:
            mp3_file = self.get_pair(file)
            if mp3_file is None:
                print(f"Pair not found for file {file}. Skipping...")
                continue
            self.data.append(file)
            self.metadata.append(mp3_file)
            if len(self.data) >= self.size: break
