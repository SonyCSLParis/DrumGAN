import torch
import numpy as np
import torch.nn.functional as F

from .eval_utils import *
from ..utils.utils import GPU_is_available
from ..eval.train_inception_model import SpectrogramInception3
from ..datasets.datamanager import AudioDataManager
from ..metrics.inception_score import InceptionScore
from ..audio_processing import Compose
from tools import list_files_abs_path, checkexists_mkdir, mkdir_in_path
from tqdm import trange
import os
from librosa.output import write_wav
import ipdb

from datetime import datetime


DEFAULT_MIDI_FILE_PATH = \
    f"{os.path.abspath(os.curdir)}/test_midi_files/prelude1.mid"


def interpolate_batch(x, y, steps):
    alpha = 0
    output = []
    for i in np.linspace(0., 1., steps, True):
        output.append(x*(1 - i) + y*i)
    return torch.stack(output)


def test(parser, visualisation=None):
    kwargs = vars(parser.parse_args())
    nsynth_path = kwargs.get('data_path')
    batch_size = 30

    model, config, model_name, path = load_model_checkp(**kwargs)
    midi_path = kwargs.pop('midi_path')
    if midi_path is None:
        midi_path = DEFAULT_MIDI_FILE_PATH
    midi_name = os.path.basename(midi_path).split('.')[0]

    output_path = os.path.join(path, "output")
    checkexists_mkdir(output_path)
    output_path = mkdir_in_path(output_path, "gen_from_midi")

    output_path = mkdir_in_path(output_path, model_name)
    model_loader = get_dummy_nsynth_loader(config, nsynth_path)
    model_postpro = model_loader.get_post_processor()
    overlap_index = int(model_loader.loader.audio_length/1.3)

    print("Loading MIDI file")
    from mido import MidiFile
    midi_file = MidiFile(midi_path)
    pitch_list = []
    pitch_cls_list = model_loader.loader.att_dict_list["pitch"]
    
    for i, track in enumerate(midi_file.tracks):
        for msg in track:

            if msg.type == "note_on":

                if msg.note in pitch_cls_list:
                    pitch_list.append(
                        pitch_cls_list.index(msg.note))
                else:
                    if msg.note > max(pitch_cls_list):
                        if msg.note - 12 in pitch_cls_list:
                            pitch_list.append(
                                pitch_cls_list.index(msg.note - 12))
                    if msg.note < min(pitch_cls_list):
                        if msg.note + 12 in pitch_cls_list:
                            pitch_list.append(
                                pitch_cls_list.index(msg.note + 12))

    output_audio = np.array([])
    print("Generating Bach's prelude...")
    pbar = trange(int(np.ceil(len(pitch_list)/batch_size)), desc="fake data IS loop")
    input_z, _ = model.buildNoiseData(batch_size, None, skipAtts=True)
    
    z = input_z[:, :-len(pitch_cls_list)].clone()
    z = interpolate_batch(z[0], z[1], steps=batch_size)
    n_interp = z.size(0)
    alpha = 0
    k = 0
    from time import time
    for j in pbar:
        input_labels = torch.LongTensor(pitch_list[j*batch_size: batch_size*(j+1)])
        input_z, _ = model.buildNoiseData(len(input_labels), inputLabels=input_labels.reshape(-1, 1), skipAtts=True)
        z_target = input_z[0, :-len(pitch_cls_list)].clone()
        input_z[:, :-len(pitch_cls_list)] = z.clone()[:len(input_labels)]
        gen_batch = model.test(input_z, getAvG=True)
        gen_raw = map(model_postpro, gen_batch)
        gen_raw = map(lambda x: np.array(x).astype(float), gen_raw)
        z = interpolate_batch(z[-1], z_target, batch_size)
        for i, g in enumerate(gen_raw):
            if i==0 and j == 0:
                output_audio = g
            else:
                output_audio = np.concatenate([output_audio, np.zeros(len(g) - overlap_index)])
                output_audio[-len(g):] += g
    # output_audio /= max(output_audio)
    # output_audio[output_audio > 1] = 1
    output_audio /= max(output_audio)
    write_wav(f'{output_path}/{midi_name}_{datetime.today().strftime("%Y_%m_%d_%H")}.wav', output_audio, 16000)

        

