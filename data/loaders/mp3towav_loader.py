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
from ..db_extractors.mp3towav import extract

import ipdb


FORMATS = ["wav", "mp3"]


def preprocess(preprocessing, data):
    return preprocessing(data)

class MP3ToWAV(AudioPairsLoader):
    def __init__(self,
                 path_wav,
                 path_mp3="",
                 **kargs):
        self.path_wav = path_wav
        self.path_mp3 = path_mp3
        AudioPairsLoader.__init__(self,
                                  data_path=path_wav,
                                  data_path2=path_mp3,
                                  **kargs)

    def load_data(self):
        self.data, self.header = \
            extract(path_wav=self.path_wav, 
                    path_mp3=self.path_mp3,
                    criteria=self.criteria)
