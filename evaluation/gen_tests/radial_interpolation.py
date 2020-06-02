import os
import torch
import ipdb
import random

from datetime import datetime
from .generation_tests import *

from utils.utils import mkdir_in_path, load_model_checkp, saveAudioBatch, get_device
from .generation_tests import StyleGEvaluationManager
from data.preprocessing import AudioProcessor
from data.loaders import get_data_loader


def radial_interpolation(z_dim: int, n_samples: int=10):
    fixed_point = torch.randn(z_dim)
    fixed_point = fixed_point / fixed_point.norm()
    z_batch = torch.zeros(n_samples, z_dim)

    t = torch.linspace(0.2, 8.0, n_samples)
    for k in range(n_samples):
        z_batch[k] = fixed_point * t[k]
    return z_batch


def generate(parser):
    args = parser.parse_args()
    device = get_device()

    model, config, model_name = load_model_checkp(**vars(args))
    latentDim = model.config.noiseVectorDim
    transform_config = config['transform_config']
    loader_config = config['loader_config']
    # We load a dummy data loader for post-processing
    processor = AudioProcessor(**transform_config)

    dbname = loader_config['dbname']
    loader_config["criteria"]["size"] = 1000
    loader = get_data_loader(dbname)(
        name=dbname + '_' + transform_config['transform'],
        preprocessing=processor, **loader_config)


    label = torch.Tensor(random.sample(loader.metadata, k=1))

    labels, _ = model.buildNoiseData(1, inputLabels=label, skipAtts=True)
    z = labels.repeat(args.n_gen, 1)

    z_noise = radial_interpolation(latentDim, args.n_gen)

    z[:, :latentDim] = z_noise

    gnet = model.getOriginalG()
    gnet.eval()
    with torch.no_grad():
        out = gnet(z.to(device)).detach().cpu()

        audio_out = loader.postprocess(out)

    # Create output evaluation dir
    output_dir = mkdir_in_path(args.dir, f"generation_tests")
    output_dir = mkdir_in_path(output_dir, model_name)
    output_dir = mkdir_in_path(output_dir, "radial_interpolation")
    output_dir = mkdir_in_path(output_dir, datetime.now().strftime('%Y-%m-%d %H:%M'))

    saveAudioBatch(audio_out,
                   path=output_dir,
                   basename='test_radial_interpolation', 
                   sr=config["transform_config"]["sample_rate"])
    print("FINISHED!\n")