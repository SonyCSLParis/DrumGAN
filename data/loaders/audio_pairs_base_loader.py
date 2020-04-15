import os
import os.path
import dill
import torch.utils.data as data
import torch
from utils.utils import mkdir_in_path, read_json, filter_keys_in_strings, list_files_abs_path, get_filename
from abc import ABC, abstractmethod

import numpy as np

from ..db_stats import buildKeyOrder
from .base_loader import AudioDataLoader
from tqdm import tqdm, trange

import logging
from random import shuffle

from copy import deepcopy
# from audiolazy.lazy_midi import midi2str

import ipdb


FORMATS = ["wav", "mp3"]

class AudioPairsLoader(AudioDataLoader, ABC):
    def __init__(self,
                 data_path2,
                 **kargs):

        self.data_path2 = data_path2
        AudioDataLoader.__init__(self, **kargs)
    # @abstractmethod
    # def get_pair(self):
    #     raise NotImplementedError

    def __getitem__(self, index):
        if self.getitem_processing:
            return self.getitem_processing(self.data_x[index]),\
                   self.getitem_processing(self.data_y[index])
        else:
            return self.data_x[index], self.data_y[index]
    
    def __len__(self):
        return len(self.data_x)

    def get_random_labels(self, batch_size):
        raise NotImplementedError(
            "AudioPairsLoader cannot load randomly generated pair")

    def preprocess_data(self):
        print("Preprocessing data...")
        import multiprocessing
        p = multiprocessing.Pool(multiprocessing.cpu_count())
        self.data_x, self.data_y = list(zip(*self.data))
        try:
            self.data_x = list(p.map(self.preprocessing,
                tqdm(self.data_x, desc='preprocessing-x')))
        except (MemoryError, OSError) as e:
            self.data_x = list(map(self.preprocessing,
                tqdm(self.data_x, desc='preprocessing-x')))
        try:
            self.data_y = list(p.map(self.preprocessing,
                tqdm(self.data_y, desc='preprocessing-y')))
        except (MemoryError, OSError) as e:
            self.data_y = list(map(self.preprocessing,
                tqdm(self.data_y, desc='preprocessing-y')))
        p.close()
        p.join()
        print("Data preprocessing done")

    def shuffle_data(self):
        shuffle(self.data)

    def index_to_labels(self, batch, transpose=False):

        labels = torch.zeros_like(batch).tolist()
        for i, att_dict in enumerate(self.header['attributes'].values()):
            for j, idx in enumerate(batch[:, i]):
                labels[j][i] = att_dict['values'][idx]
        if transpose:
            return list(zip(*labels))
        return labels

    def train_val_split(self, tr_val_split=0.9):
        assert len(self.data) > 0, "tr/val split: No loaded data yet"
        if not self.shuffle:
            print("WARNING: splitting train/val data without shuffling!")

        tr_size = int(np.floor(len(self.data) * tr_val_split))
        val_size = int(np.ceil(len(self.data) * (1 - tr_val_split)))

        self.val_data_x = self.data_x[-val_size:]
        self.val_data_y = self.data_y[-val_size:]
        self.tr_data_x = self.data_x[:tr_size]
        self.tr_data_y = self.data_y[:tr_size]

        del self.data_x
        del self.data_y
        self.data_x = self.tr_data_x
        self.data_y = self.tr_data_y
        del self.tr_data_x
        del self.tr_data_y
        del self.data

    def get_validation_set(self, batch_size=None, process=False):
        if batch_size is None:
            batch_size = len(self.val_data_x)
        val_batch_x = self.val_data_x[:batch_size]
        val_batch_y = self.val_data_y[:batch_size]
        if process:
            val_batch_x = \
                torch.stack([self.getitem_processing(v) for v in val_batch_x])
            val_batch_y = \
                torch.stack([self.getitem_processing(v) for v in val_batch_y])
        # return torch.stack(val_batch_x), torch.stack(val_batch_y)
        return torch.FloatTensor(val_batch_x), torch.FloatTensor(val_batch_y)

