from .base_loader import AudioDataLoader
from ..db_extractors.nsynth import extract


class NSynth(AudioDataLoader):
    def load_data(self):
        self.data, self.metadata, self.header = \
        	extract(self.data_path, self.criteria, download=False)
