import os
import torch
import numpy as np

from .base_loader import AudioDataLoader
from utils.utils import mkdir_in_path, read_json, filter_keys_in_strings, list_files_abs_path, get_filename
import ipdb
from ..db_extractors.mtg_drums import extract


class MTGDrums(AudioDataLoader):
    def __getitem__(self, index):
        labels = torch.Tensor(self.metadata[index]).float()
        if self.getitem_processing:
            return self.getitem_processing(self.data[index]), labels
        else:
            return self.data[index], labels

    def load_data(self):
        self.data, self.metadata, self.header = \
            extract(self.data_path, self.criteria)
    
    def index_to_labels(self, batch, transpose=False):
        if transpose:
            return list(zip(*batch))
        return batch

    # def get_validation_set(self, batch_size=None, process=False):
    #     if batch_size is None:
    #         batch_size = len(self.val_data)
    #     val_batch = self.val_data[:batch_size]

    #     val_label_batch = self.val_labels[:batch_size]
    #     if process:
    #         val_batch = \
    #             torch.stack([self.getitem_processing(v) for v in val_batch])
    #     return val_batch, torch.Tensor(val_label_batch).float()

    def get_random_labels(self, batch_size):
        return torch.rand((batch_size, len(self.header['attributes'])))

