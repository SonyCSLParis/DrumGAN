import visdom
import torch
import torchvision.transforms as Transforms
# from torchaudio.transforms import Spectrogram
import torchvision.utils as vutils

# from torchaudio import sox_effects as soxfx
# import torchaudio as taudio
import numpy as np
import random

from plotly.subplots import make_subplots
from plotly.offline import plot
import plotly.graph_objs as go
from plotly.io import write_image
import ipdb

from librosa.core import resample, stft
from librosa.display import specshow
from librosa import amplitude_to_db, magphase

from os.path import join

import matplotlib.pyplot as plt

from utils.utils import librosaSpec #, resizeAudioTensor

from .rainbowgram.wave_rain import wave2rain
from .rainbowgram.rain2graph import rain2graph

import visdom
vis = visdom.Visdom()


def save_matplot_fig(filename):
    plt.savefig(filename, dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)
    plt.close('all')

def resize_data_batch(data_batch, size):
    assert data_batch.size() == 4, "data batch has wrong dimensions"
    out_size = (data.size(0), data.size(1), size[0], size[1])

    outdata = torch.empty(out_size)
    data = torch.clamp(data_batch, min=-1, max=1)

    interpolationMode = 0
    if size[0] < data.size(0) and size[1] < data.size(1):
        interpolationMode = 2

    transform = Transforms.Compose([Transforms.Normalize((-1., -1., -1.), (2, 2, 2)),
                                    Transforms.ToPILImage(),
                                    Transforms.Resize(
                                        out_size_image, interpolation=interpolationMode),
                                    Transforms.ToTensor()])

    for i in range(out_size[0]):
        outdata[i] = transform(data[i])

    return outdata


def scatter_plotly(data, title, xtitle="time", ytitle="amplitude"):
    # assert type(data) == list, "scatter_plotly expects list as input"
    # if type(data) is not list:
    #     data = list(data)
    trace = go.Scatter(x=[i for i in range(len(data))],
                       y=data)
    fig = go.Figure(
            data=[trace],
            layout=dict(title=title, autosize=True))
    fig.update_xaxes(title_text=xtitle)
    fig.update_yaxes(title_text=ytitle)
    return fig

def rainbowgram_matplot(audio, title, figsize=(6.4, 4.8)):
    if type(audio) == torch.Tensor:
        audio = audio.numpy()
    assert type(audio) is np.ndarray, "Error matplot_rainbowgram"
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes()
    # if type(audio) is torch.Tensor:
    #     audio = audio.numpy().reshape(-1).astype(float)
    rain = wave2rain(audio, sr=16000, stride=64, log_mag=True, clip=0.1)
    ax = rain2graph(rain[:,::-1,:], ax=ax, fig=fig)
    ax.xaxis.tick_bottom()
    ax.invert_yaxis()
    ax.set_title(title)
    plt.xlabel('time')
    plt.ylabel('freq')
    fig.tight_layout()
    return fig

def plotlyHeatmap(data, title, subplot_titles=['mag', 'phase']):
    if type(data) == torch.Tensor:
        data = data.cpu().numpy()
    if data.shape[0] == 2:
        mag_spec, ph_spec = data[0], data[1]
    else:
        mag_spec, ph_spec = librosaSpec(data)
    fig = make_subplots(rows=1, cols=2, subplot_titles=subplot_titles, print_grid=False, shared_yaxes=True)
    mtrace = go.Heatmap(z=list(mag_spec), colorbar=dict(x=0.45))
    ptrace = go.Heatmap(z=list(ph_spec))
    fig.append_trace(mtrace, 1, 1)
    fig.update_xaxes(title_text="time", row=1, col=1)
    fig.update_yaxes(title_text="freq", row=1, col=1)
    fig.append_trace(ptrace, 1, 2)
    fig.update_xaxes(title_text="time", row=1, col=2)
    fig.update_layout(
        autosize=True, 
        title_text=title)
    return fig

def heatmap_plotly(hm, title, xtitle='time', ytitle='freq'):
    fig = go.Figure()
    trace = go.Heatmap(z=list(hm))
    fig.add_trace(trace)
    fig.update_xaxes(title_text=xtitle)
    fig.update_yaxes(title_text=ytitle)
    fig.update_layout(
        autosize=True, 
        title_text=title)
    return fig

def confusion_matrix_plotly(cm, title, class_list):
    fig = go.Figure()
    trace = go.Heatmap(x=class_list, y=class_list, z=cm)
    fig.add_trace(trace)
    fig.update_layout(title=title + ' confusion matrix',
                         xaxis_title='prediction',
                         yaxis_title='ground truth')
    return fig

def magph_plotly(spectrum, title, subplot_titles=['mag', 'phase']):
    assert type(spectrum) is np.ndarray, "Error magph_plotly"
    # if type(data) == torch.Tensor:
    #     data = data.numpy()
    assert spectrum.shape()[0] == 2, "Error magph_plotly"

    mag_spec, ph_spec = spectrum[0], spectrum[1]

    mag_spec, ph_spec = librosaSpec(data)
    fig = make_subplots(rows=1,
                        cols=2, 
                        subplot_titles=subplot_titles, 
                        print_grid=False, 
                        shared_yaxes=True)
    mtrace = go.Heatmap(z=list(mag_spec), 
                        colorbar=dict(x=0.45))
    ptrace = go.Heatmap(z=list(ph_spec))
    fig.append_trace(mtrace, 1, 1)
    fig.update_xaxes(title_text="time", row=1, col=1)
    fig.update_yaxes(title_text="freq", row=1, col=1)
    fig.append_trace(ptrace, 1, 2)
    fig.update_xaxes(title_text="time", row=1, col=2)
    fig.update_layout(autosize=True, title_text=title)
    return fig


def subplotScatter(data, title, maxFigs=10):
    nFigs = min(maxFigs, len(data))
    subplot_titles = [f'sample_{i}' for i in range(nFigs)]
    fig = make_subplots(rows=1, cols=len(data), subplot_titles=subplot_titles, print_grid=False)
    fig['layout'].update(title=title,
                         autosize=True
                         )
    
    for i in range(nFigs):
        y = data[i]
        ysize = len(y)
        trace = go.Scatter(x=list(range(ysize)), y=y.numpy().tolist())
        fig.append_trace(trace, 1, 1 + i)
    return fig


def subplotSpectrogram(data, title, maxFigs=5):
    nFigs = min(maxFigs, len(data))
    subplot_titles = [f'sample_{i}' for i in range(nFigs)]
    fig = make_subplots(rows=len(data), cols=2, subplot_titles=subplot_titles, print_grid=False)
    fig['layout'].update(title=title,
                         autosize=True
                         # height=4096,
                         # width=1420
                         )
    for i in range(nFigs):
        y = data[i].reshape(-1)

        resample_y = resizeAudioTensor(y, orig_sr=len(y)) # we assume is one sec --> sr=len(data)
        # mag_spec, ph_spec = librosaSpec(resample_y)

        mtrace = go.Heatmap(z=list(mag_spec))
        ptrace = go.Heatmap(z=list(ph_spec))
        fig.append_trace(mtrace, i + 1, 1)
        fig.append_trace(ptrace, i + 1, 2)
    return fig


def subplotSpecshow(data, title, maxFigs=5):
    nFigs = min(maxFigs, len(data))
    subplot_titles = [f'sample_{i}' for i in range(nFigs)]
    fig = plt.figure(title, figsize=(10, 10))
    fig.suptitle(title, fontsize=10)

    for i in range(nFigs):
        y = data[i].reshape(-1)

        # TODO: Change the orig_sr to an arg. It assumes that we are generating 1s audio
        resample_y = resizeAudioTensor(y, orig_sr=len(y)) # we assume is one sec --> sr=len(data)
        mag, ph = librosaSpec(resample_y)

        ax1 = fig.add_subplot(nFigs, 2, 2*i + 1)
        specshow(mag, sr=16000, y_axis='log', x_axis='time')
        ax2 = fig.add_subplot(nFigs, 2, 2*i + 2)
        specshow(ph, sr=16000, y_axis='log', x_axis='time')

    return fig


def saveImage(data, out_size_image, path):
    outdata = resize_data_batch(data, out_size_image)
    vutils.save_image(outdata, path)


def delete_env(name):

    vis.delete_env(name)


def publishScatterPlot(data, name="", window_token=None):
    r"""
    Draws 2D or 3d scatter plots

    Args:

        data (list of tensors): list of Ni x 2 or Ni x 3 tensors. Each tensor
                        representing a cloud of data
        name (string): plot name
        window_token (token): ID of the window the plot should be done

    Returns:

        ID of the window where the data are plotted
    """

    if not isinstance(data, list):
        raise ValueError("Input data should be a list of tensors")

    nPlots = len(data)
    colors = []

    random.seed(None)

    for item in range(nPlots):
        N = data[item].size()[0]
        colors.append(torch.randint(0, 256, (1, 3)).expand(N, 3))

    colors = torch.cat(colors, dim=0).numpy()
    opts = {'markercolor': colors,
            'caption': name}
    activeData = torch.cat(data, dim=0)

    return vis.scatter(activeData, opts=opts, win=window_token, name=name)


def saveTensor(data, out_size_image, path):

    if data.size()[2:] == out_size_image:
        outdata = data
    else:
        outdata = resizeTensor(data, out_size_image)
    vutils.save_image(outdata, path)


def resizeTensor(data, out_size_image):

    out_data_size = (data.size()[0], data.size()[
                     1], out_size_image[0], out_size_image[1])

    outdata = torch.empty(out_data_size)
    data = torch.clamp(data, min=-1, max=1)

    interpolationMode = 0
    if out_size_image[0] < data.size()[0] and out_size_image[1] < data.size()[1]:
        interpolationMode = 2

    transform = Transforms.Compose([Transforms.Normalize((-1., -1., -1.), (2, 2, 2)),
                                    Transforms.ToPILImage(),
                                    Transforms.Resize(
                                        out_size_image, interpolation=interpolationMode),
                                    Transforms.ToTensor()])

    for img in range(out_data_size[0]):
        outdata[img] = transform(data[img])

    return outdata


def publishLoss(data, name="", window_tokens=None, env="main"):

    if window_tokens is None:
        window_tokens = {key: None for key in data}

    for key, plot in data.items():

        if key in ("scale", "iter"):
            continue

        nItems = len(plot)
        inputY = np.array([plot[x] for x in range(nItems) if plot[x] is not None])
        inputX = np.array([data["iter"][x] for x in range(nItems) if plot[x] is not None])

        opts = {'title': key + (' scale %d loss over time' % data["scale"]),
                'legend': [key], 'xlabel': 'iteration', 'ylabel': 'loss'}

        window_tokens[key] = vis.line(X=inputX, Y=inputY, opts=opts,
                                      win=window_tokens[key], env=env)

    return window_tokens



################

def plot_prf(precision, recall, fscore, title):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=precision,
            name=f"precision",
            mode='lines'
        )
    )
    fig.add_trace(
        go.Scatter(
            y=recall,
            name=f"recall",
            mode='lines'
        )
    )
    fig.add_trace(
        go.Scatter(
            y=fscore,
            name=f"fscore",
            mode='lines'
        )
    )
    fig.update_layout(title=title + ' precision, recall, f-score')
    return fig

def plotly_classification_report(p, r, f, class_list, name):
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    p, r, fs, support = precision_recall_fscore_support(y_true, y_pred)
    cm = confusion_matrix(y_true, 
                          y_pred, 
                          labels=class_list)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=cls_report[name]['m_p'],
            name=f"precision",
            mode='lines'
        )
    )
    fig.add_trace(
        go.Scatter(
            y=cls_report[name]['m_r'],
            name=f"recall",
            mode='lines'
        )
    )
    fig.add_trace(
        go.Scatter(
            y=cls_report[name]['m_fs'],
            name=f"fscore",
            mode='lines'
        )
    )
    fig.update_layout(title=name)

    fig_cm = confusion_matrix_plotly(
        cm, 
        title=name, 
        class_list=att_list)

    return fig, fig_cm

