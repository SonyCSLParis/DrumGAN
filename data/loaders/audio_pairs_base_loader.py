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

    @abstractmethod
    def get_pair(self):
        raise NotImplementedError

