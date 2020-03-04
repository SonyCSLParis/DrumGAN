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
from tqdm import trange, tqdm
import os
from librosa.output import write_wav
from librosa.core import load
import ipdb


# DEFAULT_TEST_AUDIOS_PATH = "test_audios/"
SAMPLE_RATE = 16000

def test(parser, visualisation=None):
    kwargs = vars(parser.parse_args())
    nsynth_path = kwargs.get('data_path')
    scheck = kwargs.get('scheck', False)
    batch_size = 30
    n_iter = 500
    n_gens = 100
    device = 'cuda' if GPU_is_available() else 'cpu'
    model, config, model_name, path = load_model_checkp(**kwargs)

    output_path = os.path.join(path, "output")
    output_path = os.path.join(path, "EVAL")
    checkexists_mkdir(output_path)
    output_path = mkdir_in_path(output_path, "audio_to_z_optimization")
    output_path = mkdir_in_path(output_path, model_name)
    model_loader = get_dummy_nsynth_loader(config, nsynth_path)
    loader = model_loader.get_loader()
    loader = torch.utils.data.DataLoader(loader, 
                                    batch_size=1, 
                                    shuffle=True, 
                                    pin_memory=True, 
                                    num_workers=0)

    model_postpro = model_loader.get_post_processor()
    loader_iter = iter(loader)
    att_one_hot_dim = model.ClassificationCriterion.getInputDim()

    generator = model.avgG
    generator.train()
    generator.to(device)
    generator.requires_grad = True
    i = 0
    err_score = []
    pbar1 = tqdm(loader_iter, desc='Experiments loop')
    for target, label in pbar1:
        
        if scheck:
            target = generator(model.buildNoiseData(1, inputLabels=label)[0]).detach().double()
        z = model.buildNoiseData(1)[0]
        z_noise = z[:, :-att_one_hot_dim]
        ztarget = model.buildNoiseData(1, inputLabels=label)[0]
        ztarget = ztarget[:,-att_one_hot_dim:]
        ztarget.requires_grad = False
        z_noise.requires_grad = True
        optimizer = torch.optim.Adam([z_noise], lr=1e-2)

        # print(f"Initial Z:\n{torch.cat([z_noise, ztarget], dim=1)}\n")

        pbar = trange(n_iter, desc='optimization loop')
        for j in pbar:
            optimizer.zero_grad()
            in_z = torch.cat([z_noise, ztarget], dim=1)
            gen = generator(in_z.to(device))

            err = ((target - gen.double())**2).mean()
            err.backward()
            optimizer.step()
            pbar.set_description(f"Error: {err}")
            err_score.append(err.item())
        # print(f"Final Z:\n{in_z}\n")
        i+=1

        saveAudioBatch(map(model_postpro, gen.detach()), output_path, basename=f"gen_{i}")
        saveAudioBatch(map(model_postpro, target), output_path, basename=f"true_{i}")  
        pbar1.set_description(f"Mean error {np.mean(err_score)}")
        if i == n_gens:
            break
    with open(f'{output_path}/mean_error_{str(scheck)}.txt', 'w') as out_file:
        out_file.write(str(np.mean(err_score)))
        out_file.close()


