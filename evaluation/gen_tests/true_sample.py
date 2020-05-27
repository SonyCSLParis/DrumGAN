import os

from datetime import datetime
from .generation_tests import *

from utils.utils import mkdir_in_path, load_model_checkp, saveAudioBatch, read_json
from .generation_tests import StyleGEvaluationManager
from data.preprocessing import AudioProcessor
import numpy as np
import torch
from data.loaders import get_data_loader
import ipdb
import random

def generate(parser):
    parser.add_argument("--val", dest="val", action='store_true')
    parser.add_argument("-c", dest="config", type=str)
    args = parser.parse_args()

    config = read_json(args.config)
    # We load a dummy data loader for post-processing
    transform_config = config['transform_config']
    loader_config = config['loader_config']
    processor = AudioProcessor(**transform_config)
    postprocess = processor.get_postprocessor()
    assert os.path.exists(args.outdir), "Output path does not exist"
    # Create output evaluation dir
    trval = 'val' if args.val else 'train'
    output_dir = mkdir_in_path(args.outdir, f"true_sample_{config['name']}")
    output_dir = mkdir_in_path(output_dir, f"{trval}_{args.n_gen}_{datetime.now().strftime('%Y-%m-%d_%H_%M')}")

    dbname = loader_config['dbname']
    loader = get_data_loader(dbname)(
        name=dbname + '_' + transform_config['transform'],
        preprocessing=processor, **loader_config)

    if args.val:
        data, _ = loader.get_validation_set(args.n_gen)
    else:
        data = random.sample(loader.data, k=args.n_gen)
    audio_out = map(postprocess, data)
    saveAudioBatch(audio_out,
                   path=output_dir,
                   basename='true_sample',
                   sr=config["transform_config"]["sample_rate"])
    print("FINISHED!\n")