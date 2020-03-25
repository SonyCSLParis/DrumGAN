from utils.utils import list_files_abs_path, get_filename
from .base_loader import AudioDataLoader
from random import shuffle


class YouTubePianos(AudioDataLoader):
    def __init__(self, **kargs):

        AudioDataLoader.__init__(self, **kargs)

    def __getitem__(self, index):
        if self.getitem_processing:
            return self.getitem_processing(self.data[index]), -1
        else:
            return self.data[index], -1

    def read_data(self):
        files = list_files_abs_path(self.data_path, self.format)
        for file in files:
            self.data.append(file)
            if len(self.data) >= self.size: break

    def shuffle_data(self):

        shuffle(self.data)

    def get_random_labels(self, batch_size):
        return -1
    