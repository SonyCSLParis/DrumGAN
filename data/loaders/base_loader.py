import os
import os.path
import dill
import torch.utils.data as data
import torch
from utils.utils import mkdir_in_path, read_json, filter_keys_in_strings, list_files_abs_path, get_filename
from abc import ABC, abstractmethod
import numpy as np

from ..db_stats import buildKeyOrder
from tqdm import tqdm, trange

import logging
from random import shuffle

from copy import deepcopy
# from audiolazy.lazy_midi import midi2str


FORMATS = ["wav", "mp3"]

import ipdb
import hashlib


class DataLoader(ABC, data.Dataset):
    def __init__(self,
                 data_path,
                 output_path,
                 criteria,
                 getitem_processing=None,
                 dbname="default",
                 overwrite=False,
                 preprocessing=None,
                 postprocessing=None,
                 preprocess=True,
                 shuffle=False,
                 **kwargs):
        data.Dataset.__init__(self)

        # input args
        self.data_path = data_path

        self.criteria = criteria
        self.getitem_processing = getitem_processing
        self.preprocessing = preprocessing
        self.preprocess = preprocess
        self.shuffle = True
        self.postprocessing = postprocessing

        # data/metadata attributes
        # self.data, self.metadata, self.header = self.load_data()
        self.load_data()
        self.dbname = f'{dbname}_{self.__hash__()}'
        self.output_path = mkdir_in_path(os.path.expanduser(output_path), dbname)

        assert os.path.exists(self.data_path), \
            f"DataLoader error: path {self.data_path} doesn't exist"
        # assert self.format in FORMATS, \
        #     f"DataLoader error: format {self.format} not in {FORMATS}"
        self.pt_file_path = os.path.join(self.output_path, f'{self.dbname}.pt')
        # Reload dataset if exists else load and preprocess
        if os.path.exists(self.pt_file_path):
            print(f"Dataset {self.pt_file_path} exists. Reloading...")
            self.load_from_pt_file(self.pt_file_path)
        else:
            print(f"Saving dataset in {self.pt_file_path}")
            self.init_dataset()
            torch.save(self, self.pt_file_path, pickle_module=dill)
            print(f"Dataset saved.")

    def __hash__(self):
        return hex(int(self.preprocessing.__hash__(), 16) + \
               int(self.header['hash'], 16))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        labels = torch.LongTensor(self.metadata[index])
        if self.getitem_processing:
            return self.getitem_processing(self.data[index]), labels
        else:
            return self.data[index], labels


    def init_dataset(self):
        if self.shuffle: 
            self.shuffle_data()
        # preprocess data
        if self.preprocess:
            # check preprocessing is not None
            assert self.preprocessing != None, "No preprocessing was given!"
            self.preprocess_data()
        self.train_val_split()

    def load_from_pt_file(self, path):
        new_obj = torch.load(path)
        self.__dict__.update(new_obj.__dict__)

    def get_attribute_dict(self):
        return self.header['attributes']

    def get_random_labels(self, batch_size):
        labels = torch.zeros((batch_size, len(self.header['attributes'])))
        for i, att_dict in enumerate(self.header['attributes'].values()):
            labels[:, i] = torch.multinomial(
                torch.Tensor(list(att_dict['count'].values())),
                batch_size, replacement=True)
        return labels

    def preprocess_data(self):
        print("Preprocessing data...")
        import multiprocessing
        p = multiprocessing.Pool(multiprocessing.cpu_count())
        self.data = list(p.map(self.preprocessing,
                        tqdm(self.data, desc='preprocessing-loop')))
        print("Data preprocessing done")

    @abstractmethod
    def load_data(self):
        raise NotImplementedError

    def set_getitem_transform(self, transform):
        self.getitem_processing = transform

    def set_preprocessing(self, preprocessing):
        self.preprocessing = preprocessing

    def shuffle_data(self):
        combined = list(zip(self.data, self.metadata))
        shuffle(combined)
        self.data, self.metadata = zip(*combined)

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
        val_size = int(len(self.data) * (1 - tr_val_split))

        self.val_data = self.data[-val_size:]
        self.val_labels = self.metadata[-val_size:]
        self.tr_data = self.data[:tr_size]
        self.tr_labels = self.metadata[:tr_size]

        del self.data
        del self.metadata
        self.data = self.tr_data
        self.metadata = self.tr_labels
        del self.tr_data
        del self.tr_labels
        

    def get_validation_set(self, batch_size=None, process=False):
        if batch_size is None:
            batch_size = len(self.val_data)
        val_batch = self.val_data[:batch_size]
        val_label_batch = torch.LongTensor(self.val_labels[:batch_size])
        if process:
            val_batch = \
                torch.stack([self.getitem_processing(v) for v in val_batch])
        return val_batch, val_label_batch

    def postprocess(self, data_batch):
        if hasattr(self, 'post_upscale'):
            postprocess = \
                self.preprocessing.get_post_processor(self.post_upscale)
        else:
            postprocess = self.preprocessing.get_post_processor()
        return [postprocess(d) for d in data_batch]

    def get_postprocessor(self, getitem_transform=True):
        if hasattr(self, 'post_upscale'):
            return self.preprocessing.get_post_processor(self.post_upscale)
        else:
            return self.preprocessing.get_post_processor()


class AudioDataLoader(DataLoader):
    def __init__(self,
                 _format="wav",
                 **kargs):
        assert _format in ['wav', 'mp3'], f"Audio format {_format} not in wav, mp3"
        DataLoader.__init__(self, _format=_format, **kargs)
