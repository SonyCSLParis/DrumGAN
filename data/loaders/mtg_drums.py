import os
import torch
import numpy as np

from .base_loader import AudioDataLoader
from utils.utils import mkdir_in_path, read_json, filter_keys_in_strings, list_files_abs_path, get_filename


class MTGDrums(AudioDataLoader):
    ATT_LIST = [
        "duration",
        "loudness",
        "temporal_centroid",
        "log_attack_time",
        "hardness",
        "depth",
        "brightness",
        "roughness",        
        "warmth",
        "sharpness",
        "boominess"
    ]
    def __init__(self, **kargs):
        self.ATT_LIST.sort()
        self.att_dict_list = {}
        for att in self.ATT_LIST:
            self.att_dict_list[att] = ['0', '1', '2', '3', '4']
        self.att_classes = list(self.att_dict_list.values())
        AudioDataLoader.__init__(self, **kargs)

    def load_data(self):
        files = list_files_abs_path(self.data_path, 'wav')
        for file_path in files:
            file_att_path = f"{os.path.dirname(file_path)}/analysis/{get_filename(file_path)}_analysis.json"
            file_att_dict = read_json(file_att_path)

            file_metadata = []

            if any([att not in file_att_dict.keys() for att in self.ATT_LIST]):
                print(f"File {file_path} not completely annotated. Skipping...")
                continue
            for att in self.ATT_LIST:
                val = file_att_dict[att]
                if type(val) is bool:
                    file_metadata.append(1. if val else 0.)
                else:
                    file_metadata.append(val)
            if any(np.isnan(file_metadata)):
                print(f"File {file_path} contains nan values. Skipping...")
                continue
            self.metadata.append(file_metadata)
            self.data.append(file_path)
            if len(self.data) == self.size: break
        self.normalize_metadata()

    def getKeyOrders(self):
        return {att: {"order": i, "values": self.att_dict_list[att]} for i, att in enumerate(self.ATT_LIST)}

    def normalize_metadata(self):
        self.metadata = np.array(self.metadata)
        _max = np.max(self.metadata, axis=0)
        _min = np.min(self.metadata, axis=0)
        self.metadata = (self.metadata - _min) / (_max - _min)
        self.metadata = (self.metadata / 0.25).round().astype(int)

    def index_to_labels(self, index_batch):
        if type(index_batch) is np.ndarray:
            return index_batch.astype(str)
        return index_batch.numpy().astype(str)

    def __getitem__(self, index):

        if self.transform is None:
            return self.data[index], torch.LongTensor(self.metadata[index])
        else:
            return self.transform(self.data[index]), torch.LongTensor(self.metadata[index])
