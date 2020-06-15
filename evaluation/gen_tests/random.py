import os

from datetime import datetime
from .generation_tests import *

from utils.utils import mkdir_in_path, load_model_checkp, saveAudioBatch
from .generation_tests import StyleGEvaluationManager
from data.preprocessing import AudioProcessor
import numpy as np
import torch
import random
from data.loaders import get_data_loader
import ipdb
from tqdm import tqdm


def generate(parser):
    parser.add_argument("--val", dest="val", action='store_true')
    parser.add_argument("--train", dest="train", action='store_true')
    parser.add_argument("--avg-net", dest="avg_net", action='store_true')
    parser.add_argument("--name", dest="name", default="")
    parser.add_argument("--dump-labels", dest="dump_labels", action="store_true")
    args = parser.parse_args()

    model, config, model_name = load_model_checkp(**vars(args))
    latentDim = model.config.categoryVectorDim_G

    # We load a dummy data loader for post-processing
    transform_config = config['transform_config']
    loader_config = config['loader_config']
    processor = AudioProcessor(**transform_config)
    postprocess = processor.get_postprocessor()

    # Create output evaluation dir
    if args.val:
        name = args.name + '_val_labels'
    elif args.train:
        name = args.name + '_train_labels'
    else:
        name = args.name + '_rand_labels'
    if args.outdir == "":
        args.outdir = args.dir
    output_dir = mkdir_in_path(args.outdir, f"generation_samples")
    output_dir = mkdir_in_path(output_dir, model_name)
    output_dir = mkdir_in_path(output_dir, "random")
    output_dir = mkdir_in_path(output_dir, name + '_' + datetime.now().strftime('%Y-%m-%d_%H_%M'))

    dbname = loader_config['dbname']
    loader = get_data_loader(dbname)(
        name=dbname + '_' + transform_config['transform'],
        preprocessing=processor, **loader_config)

    labels = None
    if model.config.ac_gan:
        if args.val:
            val_set = loader.get_validation_set()[1]
            perm = torch.randperm(val_set.size(0))
            idx = perm[:args.n_gen]
            labels = val_set[idx]
        elif args.train:
            labels = torch.Tensor(random.sample(loader.metadata, k=args.n_gen))
        else:
            labels = loader.get_random_labels(args.n_gen)

    z, _ = model.buildNoiseData(args.n_gen, inputLabels=labels, skipAtts=True)
    data_batch = []
    with torch.no_grad():
        for i in range(int(np.ceil(args.n_gen/args.batch_size))):
            data_batch.append(
                model.test(z[i*args.batch_size:args.batch_size*(i+1)],
                toCPU=True, getAvG=args.avg_net).cpu())
        data_batch = torch.cat(data_batch, dim=0)
        audio_out = map(postprocess, data_batch)

    saveAudioBatch(audio_out,
                   path=output_dir,
                   basename='sample',
                   sr=config["transform_config"]["sample_rate"])
    if args.dump_labels:
        with open(f"{output_dir}/params_in.txt", "a") as f:
            for i in tqdm(range(args.n_gen), desc='Creating Samples'):
                params = labels[i, :-1].tolist()
                f.writelines([f"{i}, {list(params)}\n"])

    print("FINISHED!\n")