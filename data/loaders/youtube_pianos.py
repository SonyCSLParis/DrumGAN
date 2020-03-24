import os
# import os.path
# import dill
# import torch.utils.data as data
# import torch
# from utils.utils import mkdir_in_path, read_json, filter_keys_in_strings, list_files_abs_path, get_filename

# import numpy as np

# from ..db_stats import buildKeyOrder
# from tqdm import tqdm, trange

# import logging
# from random import shuffle

# from copy import deepcopy
# # from audiolazy.lazy_midi import midi2str

from .base_loader import AudioDataLoader
# FORMATS = ["wav", "mp3"]


class YouTubePianos(AudioDataLoader):
    def __init__(self, **kargs):

        AudioDataLoader.__init__(self, **kargs)

    def filter_files(self, files):
        return files[:self.size]
    