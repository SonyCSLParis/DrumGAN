import os
import os.path
import dill
import torch.utils.data as data
import torch
from utils.utils import mkdir_in_path, read_json, filter_keys_in_strings, list_files_abs_path, get_filename

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

class NSynth(AudioDataLoader):

    ATT_DICT = {
        "instrument_source": ['acoustic', 'electronic', 'synthetic'],
        "instrument_family": [
            'bass', 'brass', 'flute','guitar', 'keyboard',
            'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal'
        ],
        "instrument": [f'_{i}' for i in range(1006)],
        # "pitch":      [midi2str(i) for i in range(0, 121)],
        "velocity":   [f'_{i}' for i in range(0, 128)],
        "qualities": [
            "bright", "dark", "distortion", "fast_decay",
            "long_release", "multiphonic", "nonlinear_env",
            "percussive", "reverb", "tempo-synced"
        ]

    }
    def __init__(self,
                 attribute_file,
                 pitch_range=[24, 84],
                 **kargs):

        assert os.path.exists(attribute_file), \
            f"Metadata file {attribute_file} dosn't exist"

        self.attribute_file = attribute_file
        self.pitch_range = pitch_range

        AudioDataLoader.__init__(self, **kargs)
        
        print("N-Synth loader finished.")
        print("Attribute count:")
        print(self.att_count)
        print("")

        # free space
        del self.attribute_file

    def read_data(self):
        self.metadata_dict = read_json(self.attribute_file)

        files = list_files_abs_path(self.data_path, self.format)

        # filtered_files = self.filter_files(files)
        for file in files:
            skip = False
            file_att_dict = self.metadata_dict[get_filename(file)]
            
            # skip file if attribute not in filter attribute dict
            if self.filter_attributes != None:
                for f_att_k, f_att_val in self.filter_attributes.items():
                    assert f_att_k in file_att_dict.keys(), \
                        f"Filter attribute {f_att_k} not in {file_att_dict.keys()}."
                    if file_att_dict[f_att_k] not in f_att_val:
                        skip = True
                        break
            
            # skip files with pitch not in pitch range
            if file_att_dict['pitch'] not in range(*self.pitch_range) or skip: continue
            
            # check class balance according to balance_att
            if self.balance_att != None:
                assert self.balance_att in file_att_dict.keys(),\
                    f"Balancing attribute {self.balance_att} not in {file_att_dict.keys()}"
                assert self.balance_att in self.filter_attributes.keys(), \
                    "Balancing attribute needs to be in filter keys so as \
                    to know total number of possible values"

                att_val = file_att_dict[self.balance_att]
                if att_val not in self.att_balance_count.keys():
                    self.att_balance_count[att_val] = 0
                n_items_per_attribute = \
                    self.size / len(self.filter_attributes[self.balance_att])
                if self.att_balance_count[att_val] >= n_items_per_attribute: continue

                self.att_balance_count[att_val] += 1


            file_att_dict = self.add_item_to_attribute_value_dict(file_att_dict)
            self.data.append(file)
            self.metadata.append(file_att_dict)

            if len(self.data) >= self.size: break

        self.attribute_order = {}
        self.attribute_val_order = {}
        self.n_attributes = 0
        for att_key, att_vals in self.attribute_val_dict.items():
            self.attribute_val_dict[att_key].sort()
            if len(att_vals) == 1:
                continue

            self.attribute_order[att_key] = self.n_attributes
            self.n_attributes += 1
            self.attribute_val_order[att_key] = \
                {val: i for i, val in enumerate(self.attribute_val_dict[att_key])}

        self.count_attributes()

    def count_attributes(self):
        self.att_count = {}
        for att, vals in self.attribute_val_dict.items():
            if att not in self.att_count:
                self.att_count[att] = [0] * len(vals)
        for item_att_dict in self.metadata:
            for att, val in item_att_dict.items():
                self.att_count[att][np.where(np.array(self.attribute_val_dict[att]) == val)[0][0]] += 1
        



    def index_to_labels(self, idx_batch):
        output_labels = []
        for item_idx in idx_batch:
            item_labels = []
            idx = 0
            for att_key, att_vals in self.att_dict_list.items():
                if len(att_vals) == 1: continue
                label = self.att_dict_list[att_key][item_idx[idx]]
                item_labels.append(self.ATT_DICT[att_key][label])
                idx += 1
            output_labels.append(item_labels)
        return np.array(output_labels)

    def label_to_instrument(self, label):
        return self.ATT_DICT[int(label)]

    def label_to_source(self, label):
        return self.src_list[int(label)]
