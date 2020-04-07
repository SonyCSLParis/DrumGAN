import os
import torch
import numpy as np

from .base_loader import AudioDataLoader
from utils.utils import mkdir_in_path, read_json, filter_keys_in_strings, list_files_abs_path, get_filename
import ipdb
from ..db_extractors.mtg_drums import extract


class MTGDrums(AudioDataLoader):
    def load_data(self):
        self.data, self.metadata, self.header = \
            extract(self.data_path, self.criteria)
