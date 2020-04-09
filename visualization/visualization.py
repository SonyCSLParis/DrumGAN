import os
import numpy as np
import visdom
import ipdb


from plotly.graph_objs import Figure, Scatter
from plotly.offline import plot

from utils.utils import mkdir_in_path

from .visualization_tools import plotly_classification_report, \
    scatter_plotly, heatmap_plotly, rainbowgram_matplot, save_matplot_fig, \
    plotlyHeatmap

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from visualization.visualization_tools import confusion_matrix_plotly, plot_prf
# from .visualization_tools import *
# import plotly.tools as tls



# from librosa.util.exceptions import ParameterError
# from utils.utils import PCA
# # from ..metrics.maximum_mean_discrepancy import mmd

# import os

vis = visdom.Visdom(raise_exceptions=False)


def getVisualizer(data_type):
    return {
        # 'image': ImageVisualizer,
        'waveform': WaveformVisualizer,
        'stft':  STFTVisualizer,
        'specgrams':  SpecgramsVisualizer,
        'mel':  MelVisualizer,
        'mfcc':  MFCCVisualizer,
        'cqt': CQTVisualizer,
        'cqt_nsgt':  CQNSGTVisualizer
    }[data_type]


class TensorVisualizer(object):
    """
    """
    def __init__(self, 
                 output_path,
                 window_tokens=None,
                 env='default',
                 save_figs=True,
                 no_visdom=False,
                 **kargs):

        self.output_path = output_path
        self.window_tokens = window_tokens
        self.env = env
        self.save_figs = save_figs
        self.no_visdom = no_visdom

    def publish(self, data, name, *args):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def update_tokens(self, name):
        if self.window_tokens is None:
            self.window_tokens = {name: None}
        elif name not in self.window_tokens.keys():
            self.window_tokens[name] = None

    def publish_plotly_figure(self, fig, win, output_dir=None, env=''):
        self.update_tokens(win)
        if not self.no_visdom:
            self.window_tokens[win] = \
                vis.plotlyplot(fig, env=self.env + env, win=self.window_tokens[win])
        if output_dir:
            plot(fig, filename=os.path.join(output_dir, win + '.html'), auto_open=False)


class MetricVisualizer(TensorVisualizer):
    def __init__(self, metric_name, output_path, **kargs):
        self.metric_dict = {}
        output_path = mkdir_in_path(output_path, metric_name)
        TensorVisualizer.__init__(self, output_path=output_path, **kargs)

    # def publish(data, name):
    #     self.process_data(data, name)

    #     for key, val in self.metric_dict

class AttClassifVisualizer(TensorVisualizer):
    def __init__(self,
                 output_path,
                 attributes,
                 att_val_dict,
                 **kargs):
        # output_path = mkdir_in_path(output_path, "classification_report")
        self.attributes = attributes
        self.att_val_dict = att_val_dict
        self.metrics = {}

        TensorVisualizer.__init__(self, output_path=output_path, **kargs)

    def publish(self, true, fake, name, title):
        global vis
        if name not in self.metrics:
            self.metrics[name] = {att: {'p': [], 'r': [], 'fs':[]} for att in self.attributes}

        for i, att in enumerate(self.attributes):
            win = att + name
            self.update_tokens(win)
            p, r, fs, support = precision_recall_fscore_support(true[i], fake[i])
            cm = confusion_matrix(true[i], fake[i], labels=self.att_val_dict[att]['values'])
            fig_cm = confusion_matrix_plotly(cm, title=title + f' {att}', class_list=self.att_val_dict[att]['values'])
            self.publish_plotly_figure(fig_cm, win, output_dir=self.output_path, env='_classif')

            win += 'prfs'
            self.update_tokens(win)
            self.metrics[name][att]['p'].append(p.mean())
            self.metrics[name][att]['r'].append(r.mean())
            self.metrics[name][att]['fs'].append(fs.mean())

            fig_prf = plot_prf(self.metrics[name][att]['p'],
                     self.metrics[name][att]['r'],
                     self.metrics[name][att]['fs'],
                     title + f' {att}')

            self.publish_plotly_figure(fig_prf, win, output_dir=self.output_path, env='_classif')

    def publish_line(self, x, y, key, title):
        self.update_tokens(key)
        inputY = np.array(y)
        inputX = np.array(x)
        opts = {'title': title,
                'legend': [key], 'xlabel': 'iteration', 'ylabel': 'loss'}

        if not self.no_visdom:
            self.window_tokens[key] = vis.line(X=inputX, Y=inputY, opts=opts,
                                       win=self.window_tokens[key], env=self.env + '_loss')

        self.save(iter_n=inputX,
                loss_val=inputY,
                title=title,
                filename=os.path.join(self.output_path, f'{key}_scale_{data["scale"]}.html'))




class LossVisualizer(TensorVisualizer):
    def __init__(self,
                 output_path,
                 **kargs):
        output_path = mkdir_in_path(output_path, "loss_plots")
        TensorVisualizer.__init__(self, output_path=output_path, **kargs)

    def publish(self, data):
        global vis
        for key, _plot in data.items():
            if key in ("scale", "iter"):
              continue

            key += f"_{data['scale']}"
            self.update_tokens(key)

            nItems = len(_plot)
            inputY = np.array([_plot[x] for x in range(nItems) if _plot[x] is not None])
            inputX = np.array([data["iter"][x] for x in range(nItems) if _plot[x] is not None])

            title = key + ' scale %d loss over time' % data["scale"]
            opts = {'title': title,
                    'legend': [key], 'xlabel': 'iteration', 'ylabel': 'loss'}
            if not self.no_visdom:
                self.window_tokens[key] = vis.line(X=inputX, Y=inputY, opts=opts,
                                           win=self.window_tokens[key], env=self.env + '_loss')

            self.save(iter_n=inputX,
                    loss_val=inputY,
                    title=title,
                    filename=os.path.join(self.output_path, f'{key}_scale_{data["scale"]}.html'))



    def publish_AC_activations(self, data, axes, name="", trace_names=None, output_dir=None):
        output_dir = mkdir_in_path(output_dir, "AC_activations")
        fig = Figure()
        win = name
        data = data.numpy()
        self.update_tokens(win)
        if trace_names is None:
            trace_names = [f'trace {i}' for i in range(len(data))]
        
        for i, act in enumerate(data):
            fig.add_trace(
                    go.Scatterpolar(
                        name=str(trace_names[i]),
                        theta=axes, 
                        mode='lines',
                        r=act,
                        fill='toself'
                    )
                )

        fig.update_layout(
            title=name,
            polar=dict(
                radialaxis=dict(
                visible=True,
                range=[0, 1]
                ))
        )
        
        self.window_tokens[win] = \
            vis.plotlyplot(fig, env=self.env + '_AC', win=self.window_tokens[win])
        if output_dir:
            plot(fig, filename=join(output_dir, name + '.html'), auto_open=False)

    def publish_feature_histogram(self, data, name, env="", output_dir=None):
        self.update_tokens(name)

        fig = go.Figure()
        trace = go.Histogram(x=data, histnorm='probability density')
        fig.add_trace(trace)
        fig.update_layout(title_text=name)
        self.window_tokens[name] = vis.plotlyplot(
            fig, env=self.env + env + '_hist', win=self.window_tokens[name])
        if output_dir:
            plot(fig, filename=join(output_dir, name + '.html'), auto_open=False)

    def publish_PCA(self, data, name, total_labels, env="", labels=[], output_dir=None):
        output_dir = mkdir_in_path(output_dir, "pca")
        if not hasattr(self, 'pca_dict'):
            self.pca_dict = {}
        if len(labels) == 0:
            total_labels = ["unlabeled_trace"]
            labels = np.array(len(data) * total_labels)
        if name not in self.pca_dict:
            self.pca_dict[name] = {l: {'x': [], 'y': []} for l in total_labels}
            self.pca_dict[name]['n_steps'] = 0

        data = data.reshape(data.size(0), -1)
        pca_data = PCA(data).numpy()
        win_name = name
        self.update_tokens(win_name)

        if len(total_labels) == 0: return -1
        tr_list = []
        for label in total_labels:

            self.pca_dict[name][label]['x'].append(pca_data[labels==label, 0])
            self.pca_dict[name][label]['y'].append(pca_data[labels==label, 1])

            step_list = []
            for step in range(self.pca_dict[name]['n_steps'] + 1):
                if step < len(self.pca_dict[name][label]['x']):
                    step_list.append(go.Scatter(
                            visible = False,
                            mode='markers',
                            name = label,
                            x = self.pca_dict[name][label]['x'][step],
                            y = self.pca_dict[name][label]['y'][step]
                        ))
                else:
                    # if label is not in step add an empty trace
                    step_list.append(
                        {})

            step_list[-1]['visible'] = True

            tr_list += step_list

        fig = go.Figure(data=tr_list)
        steps = []

        for i in range(self.pca_dict[name]['n_steps'] + 1):
            step = dict(
                method = 'restyle',
                args = ['visible', [False] * len(fig.data)],
            )
            for j in range(len(total_labels)):
                step['args'][1][i + j*(self.pca_dict[name]['n_steps'] + 1)] = True # Toggle i'th trace to "visible"
            steps.append(step)
        
        sliders = [dict(
            steps = steps
        )]
        fig.update_layout(title_text=name, sliders=sliders, showlegend=True)
        self.window_tokens[win_name] = vis.plotlyplot(
            fig, env=self.env + env + '_pca', win=self.window_tokens[win_name])
        if output_dir:
            plot(fig, filename=join(output_dir, win_name + '.html'), auto_open=False)
   
        self.pca_dict[name]['n_steps'] += 1

    def publish_mmd(self, true_data, fake_data, name, _iter, output_dir=None):
        output_dir = mkdir_in_path(output_dir, "mmd")
        self.update_tokens(name)
        if not hasattr(self, 'score'):
            self.score = {}
        if name not in self.score:
            self.score[name] = {'x': [], 'y': []}
        self.score[name]['x'].append(_iter + 1)


        mmd_distance = mmd(true_data, fake_data)
        self.score[name]['y'].append(mmd_distance)
        opts = {'title': name,
                'xlabel': 'iteration', 'ylabel': 'mmd distance'}

        self.window_tokens[name] = vis.line(X=self.score[name]['x'], Y=self.score[name]['y'], opts=opts,
                                   win=self.window_tokens[name], env=self.env + '_mmd')
        if output_dir:
            self.save(iter_n=self.score[name]['x'],
                    loss_val=self.score[name]['y'],
                    title=name,
                    filename=os.path.join(output_dir, f'{name}.html'))

    def publish_fad(self, fad, name, output_dir, _iter):
        self.update_tokens(name)
        if not hasattr(self, 'score'):
            self.score = {}
        if name not in self.score:
            self.score[name] = {'x': [], 'y': []}
        self.score[name]['x'].append(_iter + 1)
        self.score[name]['y'].append(fad)

        opts = {'title': name,
                'xlabel': 'iteration', 'ylabel': 'Frèchet Audio Distance'}

        self.window_tokens[name] = vis.line(X=self.score[name]['x'], Y=self.score[name]['y'], opts=opts,
                                   win=self.window_tokens[name], env=self.env + '_fad')
        if output_dir:
            self.save(iter_n=self.score[name]['x'],
                    loss_val=self.score[name]['y'],
                    title=name,
                    filename=os.path.join(output_dir, f'{name}.html'))

    def publish_inception_score(self, iscore, name, output_dir, _iter):
        self.update_tokens(name)
        if not hasattr(self, 'score'):
            self.score = {}
        if name not in self.score:
            self.score[name] = {'x': [], 'y': []}
        self.score[name]['x'].append(_iter + 1)
        self.score[name]['y'].append(iscore)

        opts = {'title': name,
                'xlabel': 'iteration', 'ylabel': 'Inception Score'}

        self.window_tokens[name] = vis.line(X=self.score[name]['x'], Y=self.score[name]['y'], opts=opts,
                                   win=self.window_tokens[name], env=self.env + '_iscore')
        if output_dir:
            self.save(iter_n=self.score[name]['x'],
                    loss_val=self.score[name]['y'],
                    title=name,
                    filename=os.path.join(output_dir, f'{name}.html'))

    def publish_config_file(self, config_file, win='config_file'):
        self.update_tokens(win)
        output = ""
        for k, v in config_file.items():
            output+=k
            if type(v) == dict:
                output+="\n\t"
                for k1, v1 in v.items():
                    output+=f'\t{k1}: {v1}\n'
            else:
                output += f': {v}\n\n'

        self.window_tokens[win] = \
            vis.text(str(config_file), win=self.window_tokens[win], env=self.env + '_config')

    def save(self, iter_n, loss_val, title, filename):
        fig = Figure()
        trace = Scatter(x=iter_n, y=loss_val)
        fig.add_trace(trace)
        fig.layout.update(title=title)
        plot(fig, filename=filename, auto_open=False)



class AudioVisualizer(TensorVisualizer):

    MIN_SAMPLE_RATE = 8000

    def __init__(self, sampleRate, renderAudio=False, **kargs):
        self.sampleRate = sampleRate
        self.renderAudio = renderAudio
        self.max_n_plots = 10
        # TODO: parse as args following atts
        TensorVisualizer.__init__(self, **kargs)

    def get_waveform(self, x): 
        if self.postprocess:
            return self.postprocess(x)

    def set_postprocessing(self, f):
        self.postprocess = f

    def publish_audio(self, audio, win, env=''):

        if self.sampleRate >= self.MIN_SAMPLE_RATE and \
           self.renderAudio and not self.no_visdom:
            win += '_audio'
            env += '_audio'
            self.update_tokens(win)
            assert len(audio.shape) <= 2, \
            f"AudioPlayer: wrong shape {audio.shape} for audio vector"
        # HACKY: currently doing one loop per visualization/audio-player
        
            self.window_tokens[win] = \
                vis.audio(audio,
                      opts=dict(sample_frequency=self.sampleRate),
                      win=win, 
                      env=self.env + '_' + env)

    def publish_waveform(self, audio, title, win, env=''):
        win += '_wave'
        env += '_wave'
        self.update_tokens(win)
        fig_wave = scatter_plotly(audio, title=win + f'_{title}')
        if not self.no_visdom:
            self.window_tokens[win] = \
                vis.plotlyplot(fig_wave, env=self.env + '_' + env, win=win)
        if self.save:
            plot(fig_wave, filename=os.path.join(self.output_dir, win + f'_waveform_{title}.html'), auto_open=False)

    def publish_spectrogram(self, audio, title, win, env=''):
        win += '_spec'
        env += '_spec'
        self.update_tokens(win)
        fig_spec = plotlyHeatmap(audio, title=win + f'_{title}')
        if not self.no_visdom:
            self.window_tokens[win] = \
                vis.plotlyplot(fig_spec, env=self.env + '_' + env, win=win)
        if self.save:
            plot(fig_spec, filename=os.path.join(self.output_dir, win + f'_spectrum_{title}.html'), auto_open=False)

    def publish_rainbowgram(self, audio, title, win, env=''):
        win += '_rainbg'
        env += '_rainb'
        self.update_tokens(win)
        fig_rain = rainbowgram_matplot(audio, title=win + f'_{title}')
        if not self.no_visdom:
            self.window_tokens[win] =  \
                vis.matplot(fig_rain, 
                            env=self.env + '_' + env, 
                            win=win + '_rainbowgram', 
                            opts={'resizable': True})
        if self.save:
            save_matplot_fig(os.path.join(self.output_dir, win + f'_rainbowgram_{title}.png'))


class WaveformVisualizer(AudioVisualizer):
    def __init__(self, **kargs):
        AudioVisualizer.__init__(self, **kargs)

    def publish(self, data, name="", labels=[], output_dir=None):
        self.output_dir = output_dir
        n_vis = min(len(data), 3)
        for i, audio in enumerate(data[:self.max_n_plots]):
            if len(labels) != len(data):
                label_title = 'gen'
            else:
                label_title = '_'.join(labels[i])

            post_audio = self.get_waveform(audio)
            win_name = f'{name}_{str(i)}'

            # Audio player
            self.publish_audio(post_audio, win=win_name, env=str(i))
            # Audio waveform plot
            self.publish_waveform(audio.reshape(-1), title=label_title, win=win_name, env=str(i))
            # Spectrogram plot
            self.publish_spectrogram(post_audio, title=label_title, win=win_name, env=str(i))
            # Rainbowgram plot
            self.publish_rainbowgram(post_audio, title=label_title, win=win_name, env=str(i))


class STFTVisualizer(AudioVisualizer):
    def __init__(self, **kargs):
        AudioVisualizer.__init__(self, **kargs)

    def publish_complex_spec(self, audio, title, win, env):
        win += '_complex'
        env += '_complex'
        self.update_tokens(win)
        fig_spec = plotlyHeatmap(audio, title=win + f'_{title}', subplot_titles=['real', 'imaginary'])
        if not self.no_visdom:
            self.window_tokens[win] = \
                vis.plotlyplot(fig_spec, env=self.env + '_' + env, win=win)
        if self.save:
            plot(fig_spec, filename=os.path.join(self.output_dir, win + f'_complex_{title}.html'), auto_open=False)

    def publish(self, data, name="", labels=[], output_dir=None):
        self.output_dir = output_dir
        n_vis = min(len(data), 3)

        for i, audio in enumerate(data[:self.max_n_plots]):
            if len(labels) != len(data):
                label_title = 'gen'
            else:
                label_title = '_'.join(labels[i])

            post_audio = self.get_waveform(audio)

            win_name = f'{name}_{str(i)}'

            # Audio player
            self.publish_audio(post_audio, win=win_name, env=str(i))
            # Audio waveform plot
            self.publish_waveform(post_audio, title=label_title, win=win_name, env=str(i))
            # Spectrogram plot
            self.publish_spectrogram(post_audio, title=label_title, win=win_name, env=str(i))
            # Rainbowgram plot
            self.publish_rainbowgram(post_audio, title=label_title, win=win_name, env=str(i))
            # Complex plot:
            self.publish_complex_spec(audio, title=label_title, win=win_name, env=str(i))


class SpecgramsVisualizer(AudioVisualizer):
    def __init__(self, **kargs):
        AudioVisualizer.__init__(self, **kargs)

    def publish(self, data, name="", labels=[], output_dir=None):
        self.output_dir = output_dir
        # global vis
        n_vis = min(len(data), 3)

        for i, audio in enumerate(data[:self.max_n_plots]):
            if len(labels) != len(data):
                label_title = 'gen'
            else:
                label_title = '_'.join(labels[i])

            post_audio = self.get_waveform(audio)
            win_name = f'{name}_{str(i)}'

            # Audio player
            self.publish_audio(post_audio, win=win_name, env=str(i))
            # Audio waveform plot
            self.publish_waveform(post_audio, title=label_title, win=win_name, env=str(i))
            # Spectrogram plot
            self.publish_spectrogram(audio, title=label_title, win=win_name, env=str(i))
            # Rainbowgram plot
            self.publish_rainbowgram(post_audio, title=label_title, win=win_name, env=str(i))


class MelVisualizer(AudioVisualizer):
    def __init__(self, **kargs):
        AudioVisualizer.__init__(self, **kargs)

    def publish_mel_spec(self, data, title, win, env):
        win += '_mel'
        env += '_mel'
        self.update_tokens(win)
        fig_spec = heatmap_plotly(data[0], title=win + f'_{title}')
        if not self.no_visdom:
            self.window_tokens[win] = \
                vis.plotlyplot(fig_spec, env=self.env + '_' + env, win=win)
        if self.save:
            plot(fig_spec, filename=os.path.join(self.output_dir, win + f'_mel_{title}.html'), auto_open=False)


    def publish(self, data, name="", labels=[], output_dir=None):
        self.output_dir = output_dir
        # global vis
        n_vis = min(len(data), 3)

        for i, audio in enumerate(data[:self.max_n_plots]):
            if len(labels) != len(data):
                label_title = 'gen'
            else:
                label_title = '_'.join(labels[i])
            win_name = f'{name}_{str(i)}'
            try:
                post_audio = self.get_waveform(audio)
                # Audio player
                self.publish_audio(post_audio, win=win_name, env=str(i))
                # Audio waveform plot
                self.publish_waveform(post_audio, title=label_title, win=win_name, env=str(i))
                # Spectrogram plot
                self.publish_spectrogram(post_audio, title=label_title, win=win_name, env=str(i))
                # Rainbowgram plot
                self.publish_rainbowgram(post_audio, title=label_title, win=win_name, env=str(i))
                # Mel spectrogram
            except ParameterError as pe:
                print(pe)

            self.publish_mel_spec(audio, title=label_title, win=win_name, env=str(i))


class MFCCVisualizer(AudioVisualizer):
    def __init__(self, **kargs):
        AudioVisualizer.__init__(self, **kargs)

    def publish_mfcc(self, data, title, win, env):
        win += '_mfcc'
        env += '_mfcc'
        self.update_tokens(win)

        fig_spec = heatmap_plotly(data[0], title=win + f'_{title}')
        if not self.no_visdom:
            self.window_tokens[win] = \
                vis.plotlyplot(fig_spec, env=self.env + '_' + env, win=win)
        if self.save:
            plot(fig_spec, filename=os.path.join(self.output_dir, win + f'_mfcc_{title}.html'), auto_open=False)


    def publish(self, data, name="", labels=[], output_dir=None):
        self.output_dir = output_dir
        n_vis = min(len(data), 3)

        for i, audio in enumerate(data[:self.max_n_plots]):
            if len(labels) != len(data):
                label_title = 'gen'
            else:
                label_title = '_'.join(labels[i])

            win_name = f'{name}_{str(i)}'
            try:
                post_audio = self.get_waveform(audio)
                # Audio player
                self.publish_audio(post_audio, win=win_name, env=str(i))
                # Audio waveform plot
                self.publish_waveform(post_audio, title=label_title, win=win_name, env=str(i))
                # Spectrogram plot
                self.publish_spectrogram(post_audio, title=label_title, win=win_name, env=str(i))
                # Rainbowgram plot
                self.publish_rainbowgram(post_audio, title=label_title, win=win_name, env=str(i))
            except ParameterError as pe:
                print("")
                print(pe)
                print("")
            # Mel spectrogram
            self.publish_mfcc(audio, title=label_title, win=win_name, env=str(i))
        


class CQTVisualizer(AudioVisualizer):
    def __init__(self, **kargs):
        AudioVisualizer.__init__(self, **kargs)

    def publish_cqt(self, data, title, win, env):
        pass

    def publish(self, data, name="", labels=[], output_dir=None):
        self.output_dir = output_dir
        n_vis = min(len(data), 3)

        for i, audio in enumerate(data[:self.max_n_plots]):
            if len(labels) != len(data):
                label_title = 'gen'
            else:
                label_title = '_'.join(labels[i])

            post_audio = self.get_waveform(audio)
            win_name = f'{name}_{str(i)}'

            # Audio player
            self.publish_audio(post_audio, win=win_name, env=str(i))
            # Audio waveform plot
            self.publish_waveform(post_audio, title=label_title, win=win_name, env=str(i))
            # Spectrogram plot
            self.publish_spectrogram(post_audio, title=label_title, win=win_name, env=str(i))
            # Rainbowgram plot
            self.publish_rainbowgram(post_audio, title=label_title, win=win_name, env=str(i))
            # Mel spectrogram
            self.publish_spectrogram(audio, title=label_title, win=win_name, env=str(i) + '_cqt')

class CQNSGTVisualizer(AudioVisualizer):
    def __init__(self, **kargs):
        AudioVisualizer.__init__(self, **kargs)

    def publish_cqt(self, data, title, win, env):
        pass

    def publish(self, data, name="", labels=[], output_dir=None):
        self.output_dir = output_dir
        n_vis = min(len(data), 3)

        for i, audio in enumerate(data[:self.max_n_plots]):
            if len(labels) != len(data):
                label_title = 'gen'
            else:
                label_title = '_'.join(labels[i])

            post_audio = self.get_waveform(audio)
            win_name = f'{name}_{str(i)}'

            # Audio player
            self.publish_audio(post_audio, win=win_name, env=str(i))
            # Audio waveform plot
            self.publish_waveform(post_audio, title=label_title, win=win_name, env=str(i))
            # Spectrogram plot
            self.publish_spectrogram(post_audio, title=label_title, win=win_name, env=str(i))
            # Rainbowgram plot
            self.publish_rainbowgram(post_audio, title=label_title, win=win_name, env=str(i))
            # Mel spectrogram
            # self.publish_spectrogram(audio, title=label_title, win=win_name, env=str(i) + '_cqt')
