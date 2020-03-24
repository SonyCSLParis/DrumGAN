from .base_loader import AudioDataLoader
import torch
from utils.utils import mkdir_in_path
import numpy as np
from tqdm import trange


class Sinewaves(AudioDataLoader):
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
        for i in trange(self.size):
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
