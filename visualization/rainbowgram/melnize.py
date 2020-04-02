from scipy.interpolate import interp2d
import numpy as np
from toolz import curry
from librosa.core import hz_to_mel as hz2mel
from librosa.core import mel_to_hz as mel2hz


@curry
def linear2mel(spectrogram, *, freq_min, freq_max):
    """
    Convert linear-frequency into mel-frequency w/o dimension compression (with linear interpolation in linear-frequency domain)
    Args:
        spectrogram (numpy.ndarray 2D): magnitude, IF and any other mel-compatible spectrogram
    Returns:
        numpy.ndarray 2D: mel-nized spectrogram
    """
    linear_freq = np.linspace(start=freq_min, stop=freq_max, num=spectrogram.shape[0], endpoint=True)
    time = range(spectrogram.shape[1])
    melnizer = interp2d(time, linear_freq, spectrogram)

    even_spaced_mel = np.linspace(start=hz2mel(freq_min, htk=True), stop=hz2mel(freq_max, htk=True), num=spectrogram.shape[0], endpoint=True)
    mel_in_freq = [mel2hz(mel, htk=True) for mel in even_spaced_mel]
    mel_spectrogram = melnizer(time, mel_in_freq)
    return mel_spectrogram


@curry
def mel2linear(melspectrogram, *, freq_min, freq_max):
    """
    Convert mel-frequency into linear-frequency w/o dimension compression (with linear interpolation in linear-frequency domain)
    Args:
        melspectrogram (numpy.ndarray 2D): magnitude, IF and any other melspectrogram
    Returns:
        numpy.ndarray 2D: linear-nized spectrogram
    """
    time = range(melspectrogram.shape[1])
    even_spaced_mel = np.linspace(start=hz2mel(freq_min, htk=True), stop=hz2mel(freq_max, htk=True), num=melspectrogram.shape[0], endpoint=True)
    mel_in_freq = [mel2hz(mel, htk=True) for mel in even_spaced_mel]
    linearnizer = interp2d(time, mel_in_freq, melspectrogram)

    linear_freq = np.linspace(start=freq_min, stop=freq_max, num=melspectrogram.shape[0], endpoint=True)
    spectrogram = linearnizer(time, linear_freq)
    return spectrogram

scaler = 1

@curry
def linear2melD(spectrogram, *, freq_min, freq_max):
    """
    Convert linear-frequency into mel-frequency w/o dimension compression (with linear interpolation in linear-frequency domain)
    Args:
        spectrogram (numpy.ndarray 2D): magnitude, IF and any other mel-compatible spectrogram
    Returns:
        numpy.ndarray 2D: mel-nized spectrogram
    """
    linear_freq = np.linspace(start=freq_min, stop=freq_max, num=spectrogram.shape[0], endpoint=True)
    time = range(spectrogram.shape[1])
    melnizer = interp2d(time, linear_freq, spectrogram)

    even_spaced_mel = np.linspace(start=hz2mel(freq_min, htk=True), stop=hz2mel(freq_max, htk=True), num=spectrogram.shape[0]*scaler, endpoint=True)
    mel_in_freq = [mel2hz(mel, htk=True) for mel in even_spaced_mel]
    mel_spectrogram = melnizer(time, mel_in_freq)
    return mel_spectrogram


@curry
def mel2linearD(melspectrogram, *, freq_min, freq_max):
    """
    Convert mel-frequency into linear-frequency w/o dimension compression (with linear interpolation in linear-frequency domain)
    Args:
        melspectrogram (numpy.ndarray 2D): magnitude, IF and any other melspectrogram
    Returns:
        numpy.ndarray 2D: linear-nized spectrogram
    """
    time = range(melspectrogram.shape[1])
    even_spaced_mel = np.linspace(start=hz2mel(freq_min, htk=True), stop=hz2mel(freq_max, htk=True), num=melspectrogram.shape[0], endpoint=True)
    mel_in_freq = [mel2hz(mel, htk=True) for mel in even_spaced_mel]
    linearnizer = interp2d(time, mel_in_freq, melspectrogram)

    linear_freq = np.linspace(start=freq_min, stop=freq_max, num=melspectrogram.shape[0]/scaler, endpoint=True)
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
