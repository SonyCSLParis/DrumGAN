from functools import partial
from operator import pow

from scipy.interpolate import interp1d
import numpy as np
from toolz import pipe, curry
import librosa

from .melnize import linear2mel, mel2linear

    
@curry
def execf(condition: bool, f, arg):
    """Execute function "f" if "condition" is True

    Args:
        condition
    """
    return f(arg) if condition is True else arg


def wave2rain(wave, *,
    sr, n_fft: int = 1024, stride: int = 256,
    power: int = 1, clip: float = 0, log_mag: bool = False, range: bool = False, mel_freq: bool = False):
    """Convert a waveform into a rainbowgram

    Args:
        wave (numpy array [T,]): time-domain waveform
        sr:       Sampling Rate.
        n_fft:    FFT length.
        stride:   Pop length (default 3/4 overlap).
        power:    Strength coefficient (1:magnitude, 2:power).
        clip:     Magnitude clipping boarder (<=clip become clip).
        log_mag:  Flag whether use log magnitude or not.
        range:    
        mel_freq: Flag whether use mel-frequency for not.

    Returns:
        numpy.ndarray n_fft/2+1 x frame: rainbowgram
    """
    # time => time-frequency
    C = librosa.stft(wave, n_fft=n_fft, hop_length=stride)

    mag = pipe(C,
        np.abs,
        lambda mag: mag ** power,
        lambda mag: np.clip(mag, clip, None),
        execf(log_mag,  np.log10),
        # execf(mel_freq, linear2mel(freq_min=0, freq_max=sr/2))
    )

    arg = pipe(C,
        np.angle, # deg==False, (-pi, pi]
        np.unwrap,
        lambda arg: np.concatenate([arg[:,0:1], np.diff(arg, n=1)], axis=1), # [-pi, pi]
        execf(range,    interp1d([-1*np.pi-0.00001, np.pi+0.00001], [0, 1])),
        # execf(mel_freq, linear2mel(freq_min=0, freq_max=sr/2))
    )

    return np.array([mag, arg])


def rain2wave(rain, *, sr, n_fft: int = 1024, stride: int = 256, fft_configs={}, power:int = 1,
        clip:float = 0, log_mag: bool = False, range: bool = False, mel_freq: bool = False):
    """Convert a rainbowgram into a waveform

    Args:
        rain (numpy.ndarray 3D, [mag || if, freq, frame]): rainbowgram
        sr:       Sampling Rate.
        n_fft:    FFT length.
        stride:   FFT pop length (default 3/4 overlap).
        power:    Strength coefficient (1:magnitude, 2:power).
        clip:     Magnitude clipping boarder (<=clip become 0 for reconstruction).
        log_mag:  Flag whether use log magnitude or not.
        range:    
        mel_freq: Flag whether use mel-frequency for not.

    Return:
        wave (numpy array [T,]): time-domain waveform
    """
    mag = pipe(rain[0],
        # execf(mel_freq, mel2linear(freq_min=0, freq_max=sr/2)),
        execf(log_mag,  partial(pow, 10)),
        lambda mag: np.where(mag <= clip, 0, mag),
        lambda mag: mag ** (1/power),
    )

    arg = pipe(rain[1],
        # execf(mel_freq,   mel2linear(freq_min=0, freq_max=sr/2)),
        execf(range, interp1d([0, 1], [-1*np.pi-0.00001, np.pi+0.00001])),
        lambda if_gram: np.cumsum(if_gram, axis=1),
    )

    # time-frequency => time
    C = (mag + 0j) * np.exp(1j*arg)
    wave = librosa.istft(C, hop_length=stride)

    return wave
