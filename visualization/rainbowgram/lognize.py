from scipy.interpolate import interp2d
import numpy as np
from toolz import curry
from librosa.core import hz_to_mel as hz2mel
from librosa.core import mel_to_hz as mel2hz


def hz2log(hz_freq):
    pass


def log2hz(log_freq):
    pass


@curry
def linear2log(spectrogram, *, freq_min, freq_max):
    """
    Convert linear-frequency into log-frequency w/o dimension compression (with linear interpolation in linear-frequency domain)
    Args:
        spectrogram (numpy.ndarray 2D): magnitude, IF and any other log/mel-compatible spectrogram
    Returns:
        numpy.ndarray 2D: log-nized spectrogram
    """
    linear_freq = np.linspace(start=freq_min, stop=freq_max, num=spectrogram.shape[0], endpoint=True)
    time = range(spectrogram.shape[1])
    lognizer = interp2d(time, linear_freq, spectrogram)

    even_spaced_log = np.linspace(start=hz2log(freq_min), stop=hz2log(freq_max), num=spectrogram.shape[0], endpoint=True)
    log_in_freq = [log2hz(log_freq) for log_freq in even_spaced_log]
    log_spectrogram = lognizer(time, log_in_freq)
    return log_spectrogram


@curry
def log2linear(logspectrogram, *, freq_min, freq_max):
    """
    Convert mel-frequency into linear-frequency w/o dimension compression (with linear interpolation in linear-frequency domain)
    Args:
        logspectrogram (numpy.ndarray 2D): magnitude, IF and any other logspectrogram
    Returns:
        numpy.ndarray 2D: linear-nized spectrogram
    """
    time = range(logspectrogram.shape[1])
    even_spaced_log = np.linspace(start=hz2log(freq_min), stop=hz2log(freq_max), num=logspectrogram.shape[0], endpoint=True)
    log_in_freq = [log2hz(log_freq) for log_freq in even_spaced_log]
    linearnizer = interp2d(time, log_in_freq, logspectrogram)

    linear_freq = np.linspace(start=freq_min, stop=freq_max, num=logspectrogram.shape[0], endpoint=True)
    spectrogram = linearnizer(time, linear_freq)
    return spectrogram


# # 0Hz ~ 1000Hz, 200Hz/div, 5 time point
# spectrogram = [
#     [0, 1,  5, 4,  5], #    0 Hz
#     [0, 2, 10, 4, 10], #  200 Hz
#     [0, 3, 15, 4, 10], #  400 Hz
#     [0, 4, 20, 4, 30], #  600 Hz
#     [0, 5, 26, 4, 25], #  800 Hz
#     [0, 6, 30, 4, 30], # 1000 Hz
# #    0, 1, 2, 3, 4
# ]
# spectrogram_np = np.array(spectrogram)
# print(f"raw:\n{spectrogram_np}")
# mel_sp = linear2mel(spectrogram_np, freq_min=0, freq_max=1000)
# print(f"mel_sp:\n{mel_sp}")
# reconstructed = mel2linear(mel_sp, freq_min=0, freq_max=1000)
# print(f"reconstructed: \n{reconstructed}")
