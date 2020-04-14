from .base_loader import AudioDataLoader
from ..db_extractors.csl_drums import extract


class CSLDrums(AudioDataLoader):
    def load_data(self):
        self.data, self.metadata, self.header = \
            extract(self.data_path, self.criteria)
