import os
import os.path
import dill
import torch.utils.data as data
import torch
# import torchaudio
from tools import read_json, checkexists_mkdir, mkdir_in_path, filter_keys_in_strings, list_files_abs_path, get_filename

import numpy as np
import torchvision
# from torchaudio.transforms import MagPhSpectrogram

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from .db_stats import buildKeyOrder
from tqdm import tqdm, trange

import logging
from random import shuffle

from copy import deepcopy
from audiolazy.lazy_midi import midi2str


FORMATS = ["wav", "mp3"]

class DataLoader(data.Dataset):
    def __init__(self,
                 data_path,
                 output_path,
                 size,
                 _format,
                 transform=None,
                 # train_val_split=None,
                 dbname="default",
                 load_metadata=True,
                 overwrite=False,
                 preprocessing=None,
                 preprocess=True,
                 balanced_data=True,
                 shuffle=False,
                 **kargs):
        data.Dataset.__init__(self)

        self.data = []
        self.metadata = []

        self.output_path = \
            mkdir_in_path(os.path.expanduser(output_path), dbname)
        self.data_path = data_path
        self.size = size
        self.format = _format

        self.transform = transform
        # self.train_val_split = train_val_split
        self.load_metadata = load_metadata
        self.overwrite = overwrite
        self.preprocessing = preprocessing
        self._preprocess = preprocess
        self.balanced_data = balanced_data
        self.shuffle = shuffle

        assert os.path.exists(self.data_path), \
            f"DataLoader error: path {self.data_path} doesn't exist"
        assert self.format in FORMATS, \
            f"DataLoader error: format {self.format} not in {FORMATS}"

        self.pt_file_path = os.path.join(self.output_path, f'{dbname}.pt')
        # Reload dataset if exists else load and preprocess
        if os.path.exists(self.pt_file_path) and not self.overwrite:
            print(f"Dataset {self.pt_file_path} exists. Reloading...")
            # self.data, self.metadata = torch.load(self.pt_file_path)
            new_obj = torch.load(self.pt_file_path)
            self.__dict__.update(new_obj.__dict__)
        else:
            print(f"Creating dataset in {self.pt_file_path}")
            # Load data
            self.load_data()
            # Shuffle data
            if self.shuffle: self.shuffle_data()
            # Preprocessing:
            if self.preprocessing and self._preprocess: 
                self.preprocess()
            torch.save(self, self.pt_file_path, pickle_module=dill)
            print(f"Dataset saved.")
        print("Dataset loaded!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item_labels = []
        if len(self.metadata) != 0:
            for k, v in self.metadata[index].items():
                att_label = self.att_dict_list[k]
                # Skip attributes that have just one value
                if len(att_label) == 1: continue
                idx = att_label.index(v)
                item_labels.append(idx)
            item_labels = torch.LongTensor(item_labels)
        else:
            item_labels = -1
        if self.transform is not None:
            return self.transform(self.data[index]), item_labels
        else:
            return self.data[index], item_labels

    def get_labels(self, batch_size):
        return None

    def preprocess(self):
        print("Preprocessing data...")
        self.data = list(map(self.preprocessing, 
                        tqdm(self.data, desc='preprocessing-loop')))
        print("Data preprocessing done")

    def load_data(self):
        files = list_files_abs_path(self.data_path, self.format)
        filtered_files = self.filter_files(files)
        for file in filtered_files:
            self.read_item(file)

        self.sort_att_dict_list()
        self.get_att_shift_dict()


    def set_transform(self, transform):
        self.transform = transform

    def shuffle_data(self):
        combined_data = zip(self.data, self.metadata)
        shuffle(combined_data)
        self.data[:], self.metadata[:] = unzip(*shuffled_data)

    def read_item(self, item_path):
        pass

    def filter_files(self):
        raise NotImplementedError


class AudioDataLoader(DataLoader):
    def __init__(self,
                 dbname="",
                 audio_length=16000,
                 sample_rate=16000,
                 _format="wav",
                 load_metadataf=False,
                 **kargs):

        
        # audio loading parameters
        self.sample_rate = sample_rate
        self.audio_length = audio_length
        dbname += \
            f"_sr_{str(self.sample_rate)}_alen_{str(self.audio_length)}"
        DataLoader.__init__(self, dbname=dbname, _format=_format, **kargs)


class NSynthLoader(AudioDataLoader):

    ATT_DICT = {
        "instrument_source": ['acoustic', 'electronic', 'synthetic'],
        "instrument_family": [
            'bass', 'brass', 'flute','guitar', 'keyboard',
            'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal'
        ],
        "instrument": [f'_{i}' for i in range(1006)],
        "pitch":      [midi2str(i) for i in range(0, 121)],
        "velocity":   [f'_{i}' for i in range(0, 128)],
        "qualities": [
            "bright", "dark", "distortion", "fast_decay",
            "long_release", "multiphonic", "nonlinear_env",
            "percussive", "reverb", "tempo-synced"
        ]

    }
    def __init__(self,
                 att_dict_path,
                 dbname="default",
                 attribute_list=["instrument", "pitch"],
                 instrument_labels=["bass"],
                 pitch_range=[24, 84],
                 filter_keys=["acoustic"],
                 balanced_data=True,
                 **kargs):

        assert os.path.exists(att_dict_path), \
            f"Metadata file {att_dict_path} dosn't exist"

        self.instrument_labels = \
            self.ATT_DICT["instrument_family"] if \
            instrument_labels in [[], ["all"]] else \
            instrument_labels

        dbname += "_"
        for l in self.instrument_labels:
            dbname += f"{l[0]}"

        self.size = kargs.get('size', -1)
        self.attribute_list = attribute_list
        self.att_dict = read_json(att_dict_path)
        self.balanced_data = balanced_data
        self.n_items_cls = int(self.size / len(self.instrument_labels))
        self.inst_count = {k: 0 for k in self.instrument_labels}
        self.filter = filter_keys
        self.pitch_range = pitch_range
        
        # Not sure yet about these params
        self.att_dict_list = {}
        self.order_to_att = {}

        AudioDataLoader.__init__(self, dbname=dbname, **kargs)

        self.count_attributes()

        print("N-Synth loader finished.")
        print("Instrument count:")
        print(self.inst_count)
        print("")
        self.size = len(self.data)
        del self.att_dict
        self.get_att_class_list()

    def count_attributes(self):
        def count_att(att_dict):
            for att, val in att_dict.items():
                self.att_count[att][np.where(np.array(self.att_dict_list[att]) == val)[0][0]] += 1

        self.att_count = {}
        for att, vals in self.att_dict_list.items():
            if att not in self.att_count:
                self.att_count[att] = [0] * len(vals)
        list(map(lambda x: count_att(x), self.metadata))

    def train_val_split(self, tr_val_size=0.8):
        assert len(self.data) > 0, "tr/val split: No loaded data yet"
        self.indices = np.arange(len(self.data))
        shuffle(self.indices)

        self.val_size = int(self.size * (1 - tr_val_size))
        self.size = int(np.floor(self.size * tr_val_size))
        
        self.tr_indices = self.indices[:-self.val_size]
        self.val_indices = self.indices[-self.val_size + 1:]
        self.val_data = [self.data[i] for i in self.val_indices]
        self.val_labels = [self.metadata[i] for i in self.val_indices]

        self.tr_data = [self.data[i] for i in self.tr_indices]
        self.tr_labels = [self.metadata[i] for i in self.tr_indices]
        del self.data
        del self.metadata
        self.data = self.tr_data
        self.metadata = self.tr_labels

        
        for i in range(len(self.val_labels)):
            val_label = []
            for att in self.attribute_list:
                val_label.append(self.att_dict_list[att].index(self.val_labels[i][att]))
            self.val_labels[i] = val_label
        # self.val_labels = torch.from_numpy(np.array([np.where(self.instrument_labels == a)[0] for a in self.val_labels]))
        # self.val_labels = np.array([a['instrument_family'] for a in self.val_labels]).astype(int)
        return torch.stack(self.val_data), torch.Tensor(self.val_labels)

    def meets_requirements(self, att_dict):
        # TO-DO: check balance of instruments
        # Check range of pitches
        inst = att_dict['instrument_family_str']
        if inst in self.inst_count.keys() and self.inst_count[inst] <= self.n_items_cls:
            if self.pitch_range in [[], ["all"]]:
                self.inst_count[inst] += 1
                return True
            elif self.pitch_range[0] <= att_dict['pitch'] <= self.pitch_range[1] \
            and not sum(self.inst_count.values()) >= self.size:  
                self.inst_count[inst] += 1
                return True
        return False

    def get_labels(self, batch_size):
        label_batch = []
        for att, att_count in self.att_count.items():

            label_batch.append(torch.multinomial(torch.Tensor(att_count), batch_size))
        return torch.stack(label_batch, dim=1)

    def read_item(self, item_path):
        item_atts = self.att_dict[get_filename(item_path)]
        if self.meets_requirements(item_atts):
            labels = self.get_metadata(item_atts)

        # TO-DO: for the moment check_Data_balance only considers the
        # first label of the list (the instrument)
            self.data.append(item_path)
            if len(self.attribute_list) != 0:
                self.metadata.append(labels)
        else:
            pass


    def get_metadata(self, item_path):
        output_metadata = []
        specific_item_atts = self.loadAttribDict(item_path)

        return specific_item_atts

    def filter_files(self, files):
        if len(self.instrument_labels) > 0:
            audio_paths = filter_keys_in_strings(files, self.instrument_labels)
        if len(self.filter) > 0:
            audio_paths = filter_keys_in_strings(files, self.filter)
        return audio_paths


    def getKeyOrders(self, equlizationWeights=False):
        r"""
        If the dataset is labelled, give the order in which the attributes are
        given

        Returns:

            A dictionary output[key] = { "order" : int , "values" : list of
            string}
        """
        if self.att_dict_list is None:
            return None
        if equlizationWeights:
            if self.stats is None:
                raise ValueError("The weight equalization can only be \
                                 performed on labelled datasets")

            return buildKeyOrder(self.shiftAttrib,
                                 self.shiftAttribVal,
                                 stats=self.stats)
        return buildKeyOrder(self.shiftAttrib,
                             self.shiftAttribVal,
                             stats=None)

    def loadAttribDict(self,
                       item_atts):
        r"""
        Load a dictionnary describing the attributes of each image in the
        dataset and save the list of all the possible attributes and their
        acceptable values.

        Args:

            - dictPath (string): path to a json file describing the dictionnary.
                                 If None, no attribute will be loaded
            - dbDir (string): path to the directory containing the dataset
            - specificAttrib (list of string): if not None, specify which
                                               attributes should be selected
        """
        if self.attribute_list is None:
            atts = deepcopy(item_atts)
            self.attribute_list = item_atts.keys()
        else:
            atts = {k: item_atts[k] for k in self.attribute_list}

        for attribName, attribVal in atts.items():
            if attribName not in self.att_dict_list:
                self.att_dict_list[attribName] = set()
            self.att_dict_list[attribName].add(attribVal)

        # Filter the attrib list
        
        return atts

    def sort_att_dict_list(self):
        for k, v in self.att_dict_list.items():
            v = list(v)
            v.sort()
            self.att_dict_list[k] = v

    def get_att_shift_dict(self):
        self.shiftAttrib = {}
        self.shiftAttribVal = {}
        self.skipAtts = []
        self.totAttribSize = 0

        for attribName, attribVals in self.att_dict_list.items():
            if len(attribVals) == 1:
                continue

            self.shiftAttrib[attribName] = self.totAttribSize
            self.totAttribSize += 1

            self.shiftAttribVal[attribName] = {
                name: c for c, name in enumerate(attribVals)}

    def get_att_class_list(self):
        self.att_classes = []

        for k, vals in self.att_dict_list.items():
            if len(vals) == 1: continue
            self.att_classes.append([self.ATT_DICT[k][v] for v in vals])

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

    def label_to_pitch(self, label):
        return midi2str(label)


class SineWaveLoader(AudioDataLoader):
    def __init__(self,
                 freqs=[100],
                 randPh=True,
                 **kargs):
        freqs.sort()
        self.freqs = freqs
        self.randPh = randPh
        db_name = kargs.pop('db_name', 'sinewave_f')
        for f in self.freqs:
            db_name += f"_{f}"

        self.att_dict_list = freqs
        data_path = kargs.pop('data_path')
        mkdir_in_path("/tmp", "sinewaves")
        data_path = "/tmp/sinewaves/"
        print(f"Loading SineWaveLoader: {db_name}")
        AudioDataLoader.__init__(self,
                                 data_path=data_path,
                                 db_name=db_name,
                                 **kargs)

    def _gen_sinewave(self, f, ampl=1, ph=0):
        assert self.sample_rate >= 2*f, \
            "Frequency not allowed"
        return ((ampl * np.sin(
            2*np.pi * np.arange(self.audio_length)*f/self.sample_rate + ph
        )).astype(np.float32)).reshape((1, -1))

    def load_data(self):
        from librosa.output import write_wav
        for i in range(self.size):
            freq_idx = np.random.randint(len(self.freqs))
            f = self.freqs[freq_idx]
            if self.randPh:
                ph = np.random.uniform(-np.pi, np.pi)
            else:
                ph=0

            filename = f'/tmp/sinewaves/sinusoid_{str(i)}_f{str(f)}_sr{str(self.sample_rate)}.{self.format}'
            sinewave = np.array(self._gen_sinewave(f=f, ph=ph))
            self.data.append(filename)
            self.metadata.append(freq_idx)

            write_wav(filename, sinewave.reshape(-1,), self.sample_rate)
    
    def getKeyOrders(self):
        return dict(pitch=dict(order=0, values=self.freqs))

    def index_to_labels(self, index_batch):
        labels_list = []
        for ib in index_batch:
            for idx in ib:
                labels_list.append(str(self.freqs[idx]))
        return np.array(labels_list)

    def __getitem__(self, index):
        return self.transform(self.data[index]), torch.LongTensor([self.metadata[index]])

class PianosYoutube(AudioDataLoader):
    def __init__(self, path_out, **kargs):
        db_name = kargs.pop('db_name', 'youtube-pianos')
        path_to_raw = kargs.pop('path_to_raw', None)
        
        assert os.path.exists(path_to_raw), \
            "Path to raw audio does not exist"
        AudioDataLoader.__init__(self, 
                                 path_to_raw=path_to_raw, 
                                 path_out=path_out, 
                                 db_name=db_name, 
                                 **kargs)

    def _dumpDataset(self):
        """Download the yesno data if it doesn't exist in processed_folder already."""
        print('Processing...')
        audios = [x for x in os.listdir(self.path_to_raw) if self.format in x]
        print(f'Loading PianosYoutube: found {len(audios)} files in {self.format} format')
        for i, audio in enumerate(audios):
            if i == self.db_size: break

            audio_path = os.path.join(self.path_to_raw, audio)
            sig, sr = torchaudio.load(audio_path)
            sig = sig.contiguous()
            sig = sig.reshape((1, -1))
            sig = torch.FloatTensor(sig)

            self.data.append(sig)
            
        output_processed_path = os.path.join(self.path_out, self.processedDir)
        if not os.path.exists(output_processed_path): os.mkdir(output_processed_path)
        self._save_torch_file()
        print("done!")

class DrumsLoader(AudioDataLoader):
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
