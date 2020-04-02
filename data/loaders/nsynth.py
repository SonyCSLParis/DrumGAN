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
from audiolazy.lazy_midi import midi2str

import ipdb
from ..db_extractors.nsynth import extract

FORMATS = ["wav", "mp3"]

class NSynth(AudioDataLoader):
    def load_data(self):
        return extract(self.data_path, self.criteria, download=False)
