from functools import partial
import torch

from .audio_transforms import complex_to_lin, lin_to_complex, \
    RemoveDC, Compose, mag_to_complex, AddDC, safe_log_spec, \
    safe_exp_spec, mag_phase_angle, norm_audio, fold_cqt, \
    unfold_cqt, fade_out, instantaneous_freq, inv_instantanteous_freq, mel, \
    imel, reshape, to_numpy, mfcc, imfcc, cqt, icqt, loader, zeropad, stft, \
    istft

from nsgt import NSGT, LogScale, LinScale, MelScale, OctScale
import numpy as np
import librosa


import hashlib
import ipdb
# TO-DO: add function to get the output of the pipelin
# on intermediate positions

class DataProcessor(object):
    """
        This class manages all datasets and given
        the config for each of them it returns
        the corresponding DataLoader object and
        the pre-processing transforms.
    """
    # define available audio transforms
    AUDIO_TRANSFORMS = \
        ["waveform", "stft", "mel", "cqt", "cq_nsgt", "specgrams", "mfcc"]

    def __init__(self,
                 transform="waveform",
                 **kwargs):
        """
            Creates a data manager

            @arg

        """

        self.pre_pipeline = []
        self.post_pipeline = []
        self.set_atts(**kwargs)
        self.init_transform_pipeline(transform)

    def __call__(self, x):
        return self.get_preprocessor()(x)

    def __hash__(self):
        hash_list = []
        keys = list(self.__dict__.keys())
        keys.sort()
        for k in keys:
            if k not in ['pre_pipeline', 'post_pipeline']:
                hash_list.append((k, self.__dict__[k]))
        return hashlib.sha1(str(hash_list).encode()).hexdigest()

    def set_atts(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def init_transform_pipeline(self, transform):
        raise NotImplementedError

    def get_preprocessor(self, compose=True):
        if not compose:
            return self.pre_pipeline
        return Compose(self.pre_pipeline)

    def get_postprocessor(self, compose=True):
        if not compose:
            return self.post_pipeline
        return Compose(self.post_pipeline)


class AudioProcessor(DataProcessor):
    def __init__(self,
                 sample_rate=16000,
                 audio_length=16000,
                 transform='waveform',
                 **kargs):
        self.audio_length = audio_length
        self.sample_rate = sample_rate
        DataProcessor.__init__(self, transform=transform, **kargs)

    def init_transform_pipeline(self, transform):
        """
            Function that initializes the transformation pipeline

            Args:
                transform (str): name of the transformation
        """
        # Domain specific transforms
        assert transform in self.AUDIO_TRANSFORMS, \
            f"Transform '{transform}' not in {self.AUDIO_TRANSFORMS}"
        print(f"Configuring {transform} transform...")
        self.transform = transform
        {
            "waveform":  self.build_waveform_pipeline,
            "stft":      self.build_stft_pipeline,
            "specgrams": self.build_specgrams_pipeline,
            "mel":       self.build_mel_pipeline,
            "cqt":       self.build_cqt_pipeline,
            "cq_nsgt":   self.build_cqt_nsgt_pipeline,
            "mfcc":      self.build_mfcc_pipeline
        }[self.transform]()

    def build_waveform_pipeline(self):
        self._add_audio_loader()
        self._add_signal_zeropadding()
        self._add_fade_out()
        self._add_norm()
        self._add_to_torch()
        self.output_shape = (1, 1, self.audio_length)

        def in_reshape(x):
            return x.reshape(self.audio_length)

        def out_reshape(x):
            return x.reshape(self.output_shape)

        self.post_pipeline.insert(0, in_reshape)
        self.pre_pipeline.append(out_reshape)

    def build_stft_pipeline(self):
        self._init_stft_params()
        self._add_audio_loader()
        self._add_signal_zeropadding()
        self._add_fade_out()
        self._add_norm()
        self._add_stft()
        self._complex_to_lin()
        self._add_rm_dc()
        self._add_to_torch()
        self.output_shape = (2, self.n_bins, self.n_frames)

    def build_specgrams_pipeline(self):
        self._init_stft_params()
        self._add_audio_loader()
        self._add_signal_zeropadding()
        self._add_fade_out()
        self._add_norm()
        self._add_stft()
        self._add_mag_phase()
        self._add_rm_dc()
        self._add_log_mag()
        self._add_ifreq()
        self._add_to_torch()
        self.output_shape = (2, self.n_bins, self.n_frames)

    def build_mel_pipeline(self):

        self._init_stft_params()

        self.output_shape = (1, getattr(self, 'n_mels', 128), self.n_frames)

        self._add_audio_loader()
        self._add_signal_zeropadding()
        self._add_fade_out()
        self._add_norm()
        self.pre_pipeline.append(partial(mel, self.sample_rate, self.hop_size,
                                         self))
        self.post_pipeline.insert(0, partial(imel, self.sample_rate,
                                             self.hop_size, self))
        self.pre_pipeline.append(reshape)

        self.post_pipeline.insert(0, partial(reshape, self.output_shape))
        self._add_to_torch()

        # self._add_log_mag()
        self.post_pipeline.insert(0, to_numpy)

    def build_mfcc_pipeline(self):

        self._init_stft_params()
        self._add_audio_loader()
        self._add_signal_zeropadding()
        self._add_fade_out()
        self._add_norm()

        self.pre_pipeline.append(partial(mfcc, self.sample_rate, self.hop_size,
                                         self))
        self.post_pipeline.insert(0, partial(imfcc, self.sample_rate, self.hop_size,
                                         self))
        self._add_to_torch()


        self.output_shape = (1, getattr(self, 'n_mfcc', 128), self.n_frames)

        self.pre_pipeline.append(reshape)
        self.post_pipeline.insert(0, reshape)
        self.post_pipeline.insert(0, to_numpy)

    def build_cqt_pipeline(self):

        self._init_stft_params()
        self._add_audio_loader()
        self._add_signal_zeropadding()
        self._add_fade_out()
        self._add_norm()

        self.pre_pipeline.append(partial(cqt, self.sample_rate, self.hop_size,
                                         self))
        self.post_pipeline.insert(0, partial(icqt, self.sample_rate,
                                             self.hop_size))
        self._add_mag_phase()
        self._add_log_mag()
        self._add_ifreq()
        self._add_to_torch()
        self.output_shape = (2, getattr(self, 'n_cqt', 84), self.n_frames)
        self.pre_pipeline.append(reshape)
        self.post_pipeline.insert(0, reshape)

    def build_cqt_nsgt_pipeline(self):
        print("")
        print("Configuring cqt_NSGT pipeline...")

        scales = {'log': LogScale,
                  'lin': LinScale,
                  'mel': MelScale,
                  'oct': OctScale}
        nsgt_scale = scales[getattr(self, 'nsgt_scale', 'log')]
        nsgt_scale = nsgt_scale(getattr(self, 'fmin', 20),
                                getattr(self, 'fmax', self.sample_rate / 2),
                                getattr(self, 'n_bins', 96))
        nsgt = NSGT(nsgt_scale,
                    self.sample_rate,
                    self.audio_length,
                    real=getattr(self, 'real', False),
                    matrixform=getattr(self, 'matrix_form', True),
                    reducedform=getattr(self, 'reduced_form', False))
        self.n_bins = len(nsgt.wins)
        self.n_frames = nsgt.ncoefs

        self.output_shape = (2, int(self.n_bins/2), int(self.n_frames))

        self._add_audio_loader()
        self._add_signal_zeropadding()
        self._add_fade_out()
        self._add_norm()
        self.pre_pipeline.extend([lambda x: x.reshape(-1,), nsgt.forward])
        self.post_pipeline.insert(0, nsgt.backward)
        self._add_mag_phase()
        self._add_log_mag()
        self._add_ifreq()
        # Add folded cqt
        if getattr(self, 'fold_cqt', False):
            self.pre_pipeline.append(fold_cqt)
            self.post_pipeline.insert(0, unfold_cqt)
            self.output_shape = (4, int(self.n_bins/2), int(self.n_frames))


    def _add_to_torch(self):
        def to_torch(x):
            if type(x) is np.ndarray:
                return torch.from_numpy(x).float()
            else:
                return torch.FloatTensor(x)
        self.pre_pipeline.append(to_torch)

    def _add_audio_loader(self):
        self.pre_pipeline.append(partial(loader, self.sample_rate,
                                         self.audio_length))

    def _add_fade_out(self):
        # Common transforms
        if getattr(self, 'fade_out', False):
            self.pre_pipeline.append(fade_out)

    def _add_norm(self):
        if getattr(self, 'normalization', False):
            self.post_pipeline.insert(0, norm_audio)

    def _complex_to_lin(self):
        self.pre_pipeline.append(complex_to_lin)
        self.post_pipeline.insert(0, lin_to_complex)

    def _add_signal_zeropadding(self):
        self.pre_pipeline.append(partial(zeropad, self.audio_length))

    def _init_stft_params(self):
        if not hasattr(self, 'hop_size'):
            self.hop_size = int(getattr(self, 'win_size', 1024) / 2)
        if hasattr(self, 'n_frames'):
            # we substract one so we get exactly self.n_frames
            self.audio_length = self.n_frames * self.hop_size - 1

        self.n_bins = int(getattr(self, 'fft_size', 2048) / 2)

    def _add_stft(self):
        self.pre_pipeline.append(partial(stft, self))
        self.post_pipeline.insert(0, partial(istft, self))

    def _add_mag_phase(self):
        # Add complex to mag/ph transform
        self.pre_pipeline.append(mag_phase_angle)
        self.post_pipeline.insert(0, mag_to_complex)

    def _add_rm_dc(self):
        if getattr(self, 'rm_dc', True):
            # Remove DC transforms
            self.pre_pipeline.append(RemoveDC())
            self.post_pipeline.insert(0, AddDC())

    def _add_log_mag(self):
        # Log magnitude
        if getattr(self, 'log_mag', True):
            self.pre_pipeline.append(safe_log_spec)
            self.post_pipeline.insert(0, safe_exp_spec)

    def _add_ifreq(self):
        if getattr(self, 'ifreq', False):
            self.pre_pipeline.append(instantaneous_freq)
            self.post_pipeline.insert(0, inv_instantanteous_freq)

    def get_output_shape(self):
        return list(self.output_shape)

    def get_post_processor(self, insert_transform=None):
        if insert_transform is None:
            return Compose(self.post_pipeline)
        return Compose([insert_transform] + self.post_pipeline)
