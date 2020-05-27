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
from data.db_extractors.default import extract
from copy import deepcopy
# from audiolazy.lazy_midi import midi2str


FORMATS = ["wav", "mp3"]

import ipdb
import hashlib

def timeout(signum, frame):
    raise Exception("Time out!")


class DataLoader(ABC, data.Dataset):
    def __init__(self,
                 data_path,
                 criteria,
                 name,
                 output_path=None,
                 getitem_processing=None,
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

        # data/metadata/header attributes
        self.load_data()
        self.dbname = f'{name}_{self.__hash__()}'
        output_path = mkdir_in_path(self.data_path, 'processed')
        self.output_path = mkdir_in_path(os.path.expanduser(output_path), name)

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
            import joblib
            print(f"Saving dataset in {self.pt_file_path}")
            self.init_dataset()
            joblib.dump(self, self.pt_file_path)
            # torch.save(self, self.pt_file_path, pickle_module=dill)
            #print(f"Dataset saved.")

    def __hash__(self):
        return hex(int(self.preprocessing.__hash__(), 16) + \
               int(self.header['hash'], 16))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        labels = torch.Tensor(self.metadata[index])
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
        import joblib
        new_obj = joblib.load(path)
        # new_obj = torch.load(path)
        self.__dict__.update(new_obj.__dict__)

    def get_attribute_dict(self):
        return self.header['attributes']

    def get_random_labels(self, batch_size):
        labels = torch.zeros((batch_size, len(self.metadata[0])))
        shift = 0
        for i, att_dict in enumerate(self.header['attributes'].values()):
            if att_dict['type'] in [str(str), str(int)]:
                labels[:, shift] = torch.multinomial(
                    torch.Tensor(list(att_dict['count'].values())),
                    batch_size, replacement=True)
                shift += 1
            elif att_dict['type'] == str(list):
                norm_hist = torch.Tensor(list(att_dict['count'].values()))/self.header['size']
                norm_hist = norm_hist.unsqueeze(0).repeat(batch_size, 1)
                labels[:, shift: shift + len(att_dict['values'])] = \
                    torch.bernoulli(norm_hist)
                shift += len(att_dict['values'])
            elif att_dict['type'] == str(float):
                _min = torch.Tensor(list(att_dict['min'].values()))
                _max = torch.Tensor(list(att_dict['max'].values()))
                rand = torch.rand(batch_size, len(att_dict['values']))
                labels[:, shift: shift + len(att_dict['values'])] = rand
                shift += len(att_dict['values'])
        return labels

    def preprocess_data(self):
        print("Preprocessing data...")
        import multiprocessing
        import signal
        import resource

        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (32768, rlimit[1]))

        signal.signal(signal.SIGALRM, timeout)

        p = multiprocessing.Pool(multiprocessing.cpu_count())
        # signal.alarm(20)
        try:
            self.data = list(p.map(self.preprocessing,
                            tqdm(self.data, desc='preprocessing-loop')))
            p.close()
        except Exception as ex:
            print(ex)
            print("Running non-parallel processing")
            self.data = list(map(self.preprocessing,
                            tqdm(self.data, desc='preprocessing-loop')))
        # signal.alarm(0)
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
        label_batch = []
        for j, b in enumerate(batch):
            shift = 0
            labels = []
            for i, att_dict in enumerate(self.header['attributes'].values()):
                if att_dict['type'] == str(float):
                    labels += [b[shift: shift + len(att_dict['values'])].tolist()]
                    shift += len(att_dict['values'])
                elif att_dict['type'] == str(list):
                    bl = b[shift: shift + len(att_dict['values'])]
                    labels += [bl.tolist()]
                    # labels += [att_dict['values'][k] for k in np.argwhere(bl==1)[0]]                   
                    shift += len(att_dict['values'])           
                else:
                    assert b[shift] % 1 == 0, "Error in attribute orders"
                    labels += [att_dict['values'][int(b[shift])]]
                    shift += 1
            label_batch.append(labels)
        if transpose:
            return list(zip(*label_batch))
        return label_batch

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

        val_label_batch = torch.Tensor(self.val_labels[:batch_size])
        if process:
            val_batch = \
                torch.stack([self.getitem_processing(v) for v in val_batch]).float()
        else:
            val_batch = \
                torch.stack([torch.FloatTensor(v) for v in val_batch]).float()
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

class SimpleLoader(AudioDataLoader):
    def __init__(self, **kargs):
        self.metadata = []
        AudioDataLoader.__init__(self, **kargs)

    def load_data(self):
        self.data, self.header = \
            extract(self.data_path, criteria=self.criteria)
    def shuffle_data(self):
        shuffle(self.data)
    def __getitem__(self, index):
        return self.getitem_processing(self.data[index]), -1
    
    def index_to_labels(self, batch, transpose=False):
        return []