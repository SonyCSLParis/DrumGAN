from os.path import dirname, join


DEFAULT_PITCH_INCEPTION_MODEL = join(dirname(__file__), "instrument_inception_model_2020-01-17.pt")
DEFAULT_INSTRUMENT_INCEPTION_MODEL = join(dirname(__file__), "pitch_inception_model_2020-01-16.pt")

DEFAULT_INCEPTION_PREPROCESSING_CONFIG = dict(transform='specgrams',
										  	  n_frames=64,
						                      n_bins=128,
						                      fade_out=True,
						                      fft_size=2048,
						                      win_size=1024,
						                      hop_size=256,
						                      n_mel=256)