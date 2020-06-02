import torch
import numpy as np
import torch.nn.functional as F
import subprocess

# from .eval_utils import *
from utils.utils import GPU_is_available, load_model_checkp
from data.loaders import get_data_loader

from tools import list_files_abs_path, checkexists_mkdir, mkdir_in_path
from tqdm import trange, tqdm
from random import shuffle
from utils.utils import saveAudioBatch
from time import time
import os
import ipdb
from data.preprocessing import AudioProcessor

DEFAULT_INCEPTION_MODEL = \
    os.path.join(os.path.abspath(os.curdir), \
    "models/eval/inception_models/inteption_model_large_2019-11-18.pt")


def test(parser, visualisation=None):
    parser.add_argument('--device', type=str, dest='device', 
        default='cuda' if GPU_is_available() else 'cpu')
    args = parser.parse_args()

    model, config, model_name = load_model_checkp(**vars(args))
    latentDim = model.config.categoryVectorDim_G

    # We load a dummy data loader for post-processing
    transform_config = config['transform_config']
    loader_config = config['loader_config']
    processor = AudioProcessor(**transform_config)
    postprocess = processor.get_postprocessor()

    dbname = loader_config['dbname']
    loader = get_data_loader(dbname)(
        name=dbname + '_' + transform_config['transform'],
        preprocessing=processor, **loader_config)

    if args.outdir == "":
        args.outdir = args.dir
    output_path = os.path.join(args.outdir, "output")
    checkexists_mkdir(output_path)
    output_path = mkdir_in_path(output_path, model_name)
    output_path = mkdir_in_path(output_path, "gen_time")

    print("!!!!!!!STARTING GEN TIME TEST!!!!!!!!!\n")
    t0 = time()
    input_z, labels = model.buildNoiseData(args.batch_size, loader.get_random_labels(args.batch_size), skipAtts=True)
    sample_block = model.test(input_z.to(args.device), getAvG=True, toCPU=True if args.device=='cpu' else False).detach().cpu()
    gen_time = (time() - t0) * 1000 / args.batch_size
    print(f"{args.device} generation time is {gen_time} milliseconds")

    output_file = f'{output_path}/{args.device}_gen_time_{model_name}.txt'
    with open(output_file, 'w') as f:
        f.write(str(gen_time))
        f.close()

    t0 = time()
    raw_audio = list(tqdm(postprocess(sample_block), desc='Post-processing...'))
    postp_time = (time() - t0) * 1000 / args.batch_size
    print(f"{args.device} post-processing time is {postp_time} milliseconds")
    
    output_file = f'{output_path}/{device}_postpro_time_{model_name}.txt'
    with open(output_file, 'w') as f:
        f.write(str(gen_time))
        f.close()