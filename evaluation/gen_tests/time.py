import torch
import numpy as np
import torch.nn.functional as F
import subprocess

from .eval_utils import *
from ..utils.utils import GPU_is_available
from ..eval.train_inception_model import SpectrogramInception3
from ..datasets.datamanager import AudioDataManager
from ..metrics.inception_score import InceptionScore
from ..metrics.maximum_mean_discrepancy import mmd

from ..audio_processing import Compose
from tools import list_files_abs_path, checkexists_mkdir, mkdir_in_path
from tqdm import trange, tqdm
from random import shuffle
from audio.tools import saveAudioBatch
from time import time

import ipdb

DEFAULT_INCEPTION_MODEL = \
    os.path.join(os.path.abspath(os.curdir), \
    "models/eval/inception_models/inteption_model_large_2019-11-18.pt")


def test(parser, visualisation=None):

    batch_size = 20
    kwargs = vars(parser.parse_args())
    data_path = kwargs.get('data_path')
    device = 'cuda' if GPU_is_available() else 'cpu'


    model, config, model_name, path = load_model_checkp(**kwargs, useGPU=GPU_is_available())

    output_path = os.path.join(path, "output")
    checkexists_mkdir(output_path)
    output_path = mkdir_in_path(output_path, "EVAL")

    output_path = mkdir_in_path(output_path, model_name)
    output_path = mkdir_in_path(output_path, "gen_time")
    
    # GET MODEL DATA_LOADER
    data_config = config["dataConfig"]
    data_config["data_path"] = os.path.join(data_path, 'audio')
    data_config["output_path"] = "/tmp"
    data_config["loaderConfig"]["att_dict_path"] = os.path.join(data_path, 'examples.json')
    data_config["loaderConfig"]["size"] = 0
    data_config["loaderConfig"]['preprocess'] = False
    real_data_loader = AudioDataManager(**data_config, preprocess=False)
    model_postpro = real_data_loader.get_post_processor()

    print("!!!!!!!STARTING GEN TIME TEST!!!!!!!!!\n")
    t0 = time()
    input_z, labels = model.buildNoiseData(batch_size, None, skipAtts=True)
    sample_block = model.test(input_z.to(device), getAvG=True, toCPU=not GPU_is_available()).detach().cpu()
    gen_time = (time() - t0) * 1000 / batch_size
    print(f"{device} generation time is {gen_time} milliseconds")

    output_file = f'{output_path}/{device}_gen_time_{model_name}.txt'
    with open(output_file, 'w') as f:
        f.write(str(gen_time))
        f.close()

    t0 = time()
    raw_audio = list(tqdm(map(model_postpro, sample_block), desc='Post-processing...'))
    postp_time = (time() - t0) * 1000 / batch_size
    print(f"{device} post-processing time is {postp_time} milliseconds")
    
    output_file = f'{output_path}/{device}_postpro_time_{model_name}.txt'
    with open(output_file, 'w') as f:
        f.write(str(gen_time))
        f.close()