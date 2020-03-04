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

    output_path = os.path.join(path, "output")
    checkexists_mkdir(output_path)
    output_path = mkdir_in_path(output_path, "noise_injection_tests")
    output_path = mkdir_in_path(output_path, model_name)
    model_loader = get_dummy_nsynth_loader(config, nsynth_path)
    model_postpro = model_loader.get_post_processor()

    for p in model.ClassificationCriterion.inputDict['pitch']['values']:
        pitch_output_dir = mkdir_in_path(output_path, f"pitch_{str(p)}")
        z = model.buildNoiseDataWithConstraints(1, {'pitch': p})
        z = z.reshape((1, -1))
        nScales = model.netG.nScales
        noise_dim = [(1, 1, h, w) for h, w in model.netG.outputSizes]
        print("Ascending...")
        
        gen_batch = []
        noise = [torch.zeros(d) for d in noise_dim]
        gen_batch.append(model.test(z, 
                                    toCPU=False if GPU_is_available() else True, 
                                    getAvG=True,
                                    noise=noise,
                                    mean_style=None,
                                    style_weight=0,
                                    mixing_range=(-1, -1)))
        for i in range(nScales):
            noise = [torch.zeros(d) for d in noise_dim]
            for j in range(3):
                if i == 0 or i == 1:
                    noise[i] = torch.randn(noise_dim[i]) * 30
                elif i != nScales - 1:
                    noise[i] = torch.randn(noise_dim[i]) * 10
                else:
                    noise[i] = torch.randn(noise_dim[i])
                gen_batch.append(model.test(z,
                                        noise=noise,
                                        toCPU=False if GPU_is_available() else True, 
                                        getAvG=True,
                                        mean_style=None,
                                        style_weight=0))

        gen_batch = torch.cat(gen_batch, dim=0)
        saveAudioBatch(map(model_postpro, gen_batch), pitch_output_dir, basename=f'noise_inj')
    
    for p in model.ClassificationCriterion.inputDict['pitch']['values']:
        pitch_output_dir = mkdir_in_path(output_path, f"pitch_{str(p)}")
        z = model.buildNoiseDataWithConstraints(1, {'pitch': p})
        z = z.reshape((1, -1))
        nScales = model.netG.nScales
        noise_dim = [(1, 1, h, w) for h, w in model.netG.outputSizes]
        print("Ascending...")
        
        gen_batch = []
        noise = [torch.zeros(d) for d in noise_dim]
        gen_batch.append(model.test(z, 
                                    toCPU=False if GPU_is_available() else True, 
                                    getAvG=True,
                                    noise=noise,
                                    mean_style=None,
                                    style_weight=0,
                                    mixing_range=(-1, -1)))
        for i in range(nScales):
            # noise = [torch.zeros(d) for d in noise_dim]
            for j in range(3):
                # if i == 0 or i == 1:
                #     noise[i] = torch.randn(noise_dim[i]) * 50
                # elif i != nScales - 1:
                #     noise[i] = torch.randn(noise_dim[i]) * 15
                # else:
                    
                noise[i] = torch.randn(noise_dim[i]) * 10
                gen_batch.append(model.test(z,
                                        noise=noise,
                                        toCPU=False if GPU_is_available() else True, 
                                        getAvG=True,
                                        mean_style=None,
                                        style_weight=0))

        gen_batch = torch.cat(gen_batch, dim=0)
        saveAudioBatch(map(model_postpro, gen_batch), pitch_output_dir, basename=f'noise_inj_ascending')

    for p in model.ClassificationCriterion.inputDict['pitch']['values']:
        pitch_output_dir = mkdir_in_path(output_path, f"pitch_{str(p)}")
        z = model.buildNoiseDataWithConstraints(1, {'pitch': p})
        z = z.reshape((1, -1))
        nScales = model.netG.nScales
        noise_dim = [(1, 1, h, w) for h, w in model.netG.outputSizes]
        print("Ascending...")
        
        gen_batch = []
        noise = [torch.zeros(d) for d in noise_dim]
        gen_batch.append(model.test(z, 
                                    toCPU=False if GPU_is_available() else True, 
                                    getAvG=True,
                                    noise=noise,
                                    mean_style=None,
                                    style_weight=0,
                                    mixing_range=(-1, -1)))
        for i in reversed(range(nScales)):
            # noise = [torch.zeros(d) for d in noise_dim]
            for j in range(3):
                # if i == 0 or i == 1:
                #     noise[i] = torch.randn(noise_dim[i]) * 50
                # elif i != nScales - 1:
                #     noise[i] = torch.randn(noise_dim[i]) * 15
                # else:
                    
                noise[i] = torch.randn(noise_dim[i]) * 10
                gen_batch.append(model.test(z,
                                        noise=noise,
                                        toCPU=False if GPU_is_available() else True, 
                                        getAvG=True,
                                        mean_style=None,
                                        style_weight=0))

        gen_batch = torch.cat(gen_batch, dim=0)
        saveAudioBatch(map(model_postpro, gen_batch), pitch_output_dir, basename=f'noise_inj_descending')