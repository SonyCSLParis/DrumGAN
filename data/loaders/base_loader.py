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
                 size,
                 _format,
                 getitem_processing=None,
                 attributes=[],
                 filter_attributes={},
                 dbname="default",
                 load_metadata=True,
                 overwrite=False,
                 preprocessing=None,
                 preprocess=True,
                 balance_att=None,
                 shuffle=False,
                 **kwargs):
        data.Dataset.__init__(self)

        # input args
        self.data_path = data_path
        self.size = size
        self.format = _format
        self.attributes = attributes
        self.attributes.sort()
        self.filter_attributes = filter_attributes
        self.load_metadata = load_metadata
        self.overwrite = overwrite
        self.getitem_processing = getitem_processing
        self.preprocessing = preprocessing
        self.preprocess = preprocess
        self.balance_att = balance_att
        self.shuffle = shuffle

        # data/metadata attributes
        self.data = []
        self.metadata = []
        self.attributes = attributes # list of attributes 
        self.attribute_val_dict = {}
        self.att_balance_count = {}
        self.attribute_count = {}
        
        self.dbname = f'{dbname}_{self.__hash__()}'
        self.output_path = mkdir_in_path(os.path.expanduser(output_path), dbname)
        assert os.path.exists(self.data_path), \
            f"DataLoader error: path {self.data_path} doesn't exist"
        assert self.format in FORMATS, \
            f"DataLoader error: format {self.format} not in {FORMATS}"

        self.pt_file_path = os.path.join(self.output_path, f'{self.dbname}.pt')
        # Reload dataset if exists else load and preprocess
        if os.path.exists(self.pt_file_path) and not self.overwrite:
            print(f"Dataset {self.pt_file_path} exists. Reloading...")
            self.load_from_pt_file(self.pt_file_path)
        else:
            print(f"Saving dataset in {self.pt_file_path}")
            self.init_dataset()
            torch.save(self, self.pt_file_path, pickle_module=dill)
            print(f"Dataset saved.")

        if self.balance_att:
            print("")
            print(f"Balanced dataset according to {self.balance_att}")
            print(self.att_balance_count)

    def __hash__(self):
        val_list = []
        for att, val in self.__dict__.items():

            if type(val) in [str, bool, int, list]:
                val_list.append(val)
            elif type(val) is dict:
                val_list.append(val.items())
            elif hasattr(val, '__dict__'):
                for att2, val2 in val.__dict__.items():
                    if type(val2) in [str, bool, int, tuple]:
                        val_list.append(val2)
        return hashlib.sha1(str(val_list).encode()).hexdigest()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        index_labels = self.label_to_index(self.metadata[index])
        if self.getitem_processing:
            return self.getitem_processing(self.data[index]), index_labels
        else:
            return self.data[index], index_labels

    def label_to_index(self, item_dict):
        if len(self.metadata) != 0:
            item_labels = []
            for att, vals in self.attribute_val_dict.items():
                item_labels.append(vals.index(item_dict[att]))
            return torch.LongTensor(item_labels)
        else:
            return None

    def init_dataset(self):
        # read data
        self.read_data()
        # shuffle data
        if self.shuffle: 
            self.shuffle_data()
        # preprocess data
        if self.preprocess:
            # check preprocessing is not None
            assert self.preprocessing != None, "No preprocessing was given!"
            self.preprocess_data()

    def load_from_pt_file(self, path):
        new_obj = torch.load(path)
        self.__dict__.update(new_obj.__dict__)

    def get_random_labels(self, batch_size):
        label_batch = []
        for att, att_count in self.att_count.items():
            label_batch.append(torch.multinomial(torch.Tensor(att_count), batch_size))
        return torch.stack(label_batch, dim=1)

    def preprocess_data(self):
        print("Preprocessing data...")
        self.data = list(map(self.preprocessing, 
                        tqdm(self.data, desc='preprocessing-loop')))
        print("Data preprocessing done")

    @abstractmethod
    def read_data(self):
        raise NotImplementedError

    def set_getitem_transform(self, transform):
        self.getitem_processing = transform

    def set_preprocessing(self, preprocessing):
        self.preprocessing = preprocessing

    def shuffle_data(self):
        combined_data = list(zip(self.data, self.metadata))
        shuffle(combined_data)
        self.data[:], self.metadata[:] = zip(*combined_data)

    def count_attributes(self):
        self.att_count = {}
        for att, vals in self.attribute_val_dict.items():
            if att not in self.att_count:
                self.att_count[att] = [0] * len(vals)
        for item_att_dict in self.metadata:
            for att, val in item_att_dict.items():
                self.att_count[att][np.where(np.array(self.attribute_val_dict[att]) == val)[0][0]] += 1

    def add_item_to_attribute_value_dict(self, item_atts):
        r"""
        Given a dictionnary describing the attributes of a single metadata instance in the
        dataset, add it to a dictionary of all the possible attributes and their
        acceptable values. Returns a filtered dictionary of that instance with the 
        attributes of interest.

        Args:

            - dictPath (string): path to a json file describing the dictionnary.
                                 If None, no attribute will be loaded
            - dbDir (string): path to the directory containing the dataset
            - specificAttrib (list of string): if not None, specify which
                                               attributes should be selected
        """
        if len(self.attributes) == 0:
            atts = deepcopy(item_atts)
            self.attributes = item_atts.keys()
        else:
            atts = {k: item_atts[k] for k in self.attributes}

        for attribName, attribVal in atts.items():
            if attribName not in self.attribute_val_dict:
                self.attribute_val_dict[attribName] = []
            if attribVal not in self.attribute_val_dict[attribName]:
                self.attribute_val_dict[attribName].append(attribVal)
        
        return atts

    def getKeyOrders(self, equlizationWeights=False):
        r"""
        If the dataset is labelled, give the order in which the attributes are
        given

        Returns:

            A dictionary output[key] = { "order" : int , "values" : list of
            string}
        """
        if self.attribute_val_dict is None:
            return None
        if equlizationWeights:
            if self.stats is None:
                raise ValueError("The weight equalization can only be \
                                 performed on labelled datasets")

            return buildKeyOrder(self.attribute_order,
                                 self.attribute_val_order,
                                 stats=self.stats)
        return buildKeyOrder(self.attribute_order,
                             self.attribute_val_order,
                             stats=None)

    def index_to_labels(self, idx_batch):
        output_labels = []
        for item in idx_batch:
            item_labels = []
            for i, att in enumerate(self.attributes):
                if len(self.att_dict_list[att]) == 1: continue
                label = self.att_dict_list[att][item[i]]
                item_labels.append(label)
            output_labels.append(item_labels)
        return np.array(output_labels)

    def train_val_split(self, tr_val_split=0.8):
        assert len(self.data) > 0, "tr/val split: No loaded data yet"
        if not self.shuffle:
            print("WARNING: splitting train/val data without shuffling!")

        tr_size = int(np.floor(self.size * tr_val_split))
        val_size = int(self.size * (1 - tr_val_split))

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

        # convert to one hot
        for i in range(val_size):
            val_label = []
            for att in self.attributes:
                val_label.append(self.attribute_val_dict[att].index(self.val_labels[i][att]))
            self.val_labels[i] = val_label
        
    def get_val_data(self):
        return torch.stack(self.val_data), torch.LongTensor(self.val_labels)


class AudioDataLoader(DataLoader):
    def __init__(self,
                 _format="wav",
                 **kargs):
        assert _format in ['wav', 'mp3'], f"Audio format {_format} not in wav, mp3"
        DataLoader.__init__(self, _format=_format, **kargs)
