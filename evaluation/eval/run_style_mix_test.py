import torch
import numpy as np
import torch.nn.functional as F

from .eval_utils import *
from audio.tools import saveAudioBatch
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

DEFAULT_MIDI_FILE_PATH = \
    f"{os.path.abspath(os.curdir)}/test_midi_files/prelude1.mid"

def test_all_scale_gen(self, 
                         model, 
                         source: torch.Tensor, 
                         target: torch.Tensor, 
                         scale_range: tuple=(), 
                         pitch: int=55):

    gen_batch.append(self.model.test([z, mean_z], 
                                    toCPU=True, 
                                    getAvG=True,
                                    mean_style=0,
                                    style_weight=1,
                                    mixing_range=scale_range))

    return torch.cat(gen_batch)


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
    assert "StyleGAN" in model_name, f"Model {model_name} is not StyleGAN"
    midi_path = kwargs.pop('midi_path', DEFAULT_MIDI_FILE_PATH)

    output_path = os.path.join(path, "output")
    checkexists_mkdir(output_path)
    output_path = mkdir_in_path(output_path, "scale_features_tests")
    output_path = mkdir_in_path(output_path, model_name)
    model_loader = get_dummy_nsynth_loader(config, nsynth_path)
    model_postpro = model_loader.get_post_processor()
    overlap_index = int(model_loader.loader.audio_length/1.3)


    for p in model.ClassificationCriterion.inputDict['pitch']['values']:
        pitch_output_dir = mkdir_in_path(output_path, f"pitch_{str(p)}")
        z_st = model.buildNoiseDataWithConstraints(2, {'pitch': p})
        z_st = z_st.reshape(2, -1)
        zsource = z_st[0:1]
        ztarget = z_st[1:]
        nScales = model.netG.nScales

        print("Ascending...")
        gen_batch = []
        gen_batch.append(model.test([zsource], 
                                         toCPU=False if GPU_is_available() else True, 
                                         getAvG=True,
                                         mean_style=None,
                                         style_weight=1,
                                         mixing_range=(-1, -1)))
        for i in range(model.netG.nScales - 1):
            scale_range = (0, i+1)
            gen_batch.append(model.test([zsource, ztarget], 
                                         toCPU=False if GPU_is_available() else True, 
                                         getAvG=True,
                                         mean_style=None,
                                         style_weight=1,
                                         mixing_range=scale_range))
        gen_batch.append(model.test([ztarget], 
                                         toCPU=False if GPU_is_available() else True, 
                                         getAvG=True,
                                         mean_style=None,
                                         style_weight=1,
                                         mixing_range=(-1, -1)))
        gen_batch = torch.cat(gen_batch, dim=0)
        saveAudioBatch(map(model_postpro, gen_batch), pitch_output_dir, basename=f'ascending_mix')
        
        print("Descending...")
        gen_batch = []
        gen_batch.append(model.test([zsource], 
                                         toCPU=False if GPU_is_available() else True, 
                                         getAvG=True,
                                         mean_style=None,
                                         style_weight=1,
                                         mixing_range=(-1, -1)))
        for i in range(nScales - 1):
            scale_range = (nScales - i - 1, nScales)
            gen_batch.append(model.test([zsource, ztarget], 
                                         toCPU=False if GPU_is_available() else True, 
                                         getAvG=True,
                                         mean_style=None,
                                         style_weight=1,
                                         mixing_range=scale_range))
        gen_batch.append(model.test([ztarget], 
                                         toCPU=False if GPU_is_available() else True, 
                                         getAvG=True,
                                         mean_style=None,
                                         style_weight=1,
                                         mixing_range=(-1, -1)))
        gen_batch = torch.cat(gen_batch, dim=0)
        saveAudioBatch(map(model_postpro, gen_batch), pitch_output_dir, basename=f'descending_mix')

        gen_batch = []
        gen_batch.append(model.test([zsource], 
                                         toCPU=False if GPU_is_available() else True, 
                                         getAvG=True,
                                         mean_style=None,
                                         style_weight=1,
                                         mixing_range=(-1, -1)))
        for i in range(model.netG.nScales - 1):
            scale_range = (i, i+1)
            gen_batch.append(model.test([zsource, ztarget], 
                                         toCPU=False if GPU_is_available() else True, 
                                         getAvG=True,
                                         mean_style=None,
                                         style_weight=1,
                                         mixing_range=scale_range))
        gen_batch.append(model.test([ztarget], 
                                         toCPU=False if GPU_is_available() else True, 
                                         getAvG=True,
                                         mean_style=None,
                                         style_weight=1,
                                         mixing_range=(-1, -1)))
        gen_batch = torch.cat(gen_batch, dim=0)
        saveAudioBatch(map(model_postpro, gen_batch), pitch_output_dir, basename=f'scale_')
        



    print("Loading MIDI file")
    from mido import MidiFile
    midi_file = MidiFile(midi_path)
    pitch_list = []
    pitch_cls_list = model_loader.loader.att_dict_list["pitch"]
    
    for i, track in enumerate(midi_file.tracks):
        for msg in track:

            if msg.type == "note_on" and msg.note in pitch_cls_list:
                pitch_list.append(
                    pitch_cls_list.index(msg.note))

    output_audio = np.array([])
    print("Generating Bach's prelude...")
    pbar = trange(int(len(pitch_list)/batch_size), desc="fake data IS loop")
    input_z, _ = model.buildNoiseData(batch_size, None, skipAtts=True)
    
    z = input_z[:, :-len(pitch_cls_list)].clone()
    z = interpolate_batch(z[0], z[1], steps=batch_size)
    n_interp = z.size(0)
    alpha = 0
    k = 0
    from time import time
    for j in pbar:
        input_labels = torch.LongTensor(pitch_list[j*batch_size: batch_size*(j+1)])
        input_z, _ = model.buildNoiseData(batch_size, inputLabels=input_labels.reshape(-1, 1), skipAtts=True)
        z_target = input_z[0, :-len(pitch_cls_list)].clone()
        input_z[:, :-len(pitch_cls_list)] = z.clone()
        gen_batch = model.test(input_z, getAvG=True)
        gen_raw = map(model_postpro, gen_batch)
        gen_raw = map(lambda x: np.array(x), gen_raw)
        z = interpolate_batch(z[-1], z_target, batch_size)
        for i, g in enumerate(gen_raw):
            if i==0 and j == 0:
                output_audio = g
            else:
                output_audio = np.concatenate([output_audio, np.zeros(len(g) - overlap_index)])
                output_audio[-len(g):] += g
    write_wav(f'{output_path}/bach_prelude_ahasa.wav', output_audio, 16000)

        

