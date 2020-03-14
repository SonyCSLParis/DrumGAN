import torch
import numpy as np
import torch.nn.functional as F

from utils.utils import *
from datetime import datetime
from evaluation.train_inception_model import SpectrogramInception3
from .maximum_mean_discrepancy import mmd

from data.preprocessing import AudioPreprocessor
from ..inception_models import DEFAULT_PITCH_INCEPTION_MODEL, DEFAULT_INCEPTION_PREPROCESSING_CONFIG
from tqdm import trange


def test(parser, visualisation=None):
    args = parser.parse_args()

    kwargs = vars(args)
    nsynth_path = kwargs.get('data_path')
    att = kwargs.get('att_name', 'pitch')
    batch_size = kwargs.get('batch_size', 50)
    is_samples = kwargs.get('is_samples', 5000)

    true_files = list_files_abs_path(args.true_path, 'wav')
    fake_files = list_files_abs_path(args.fake_path, 'wav')

    n_samples = min(len(true_files), len(fake_files))
    is_samples = min(n_samples, is_samples)

    if args.inception_model == None:
        args.inception_model = DEFAULT_PITCH_INCEPTION_MODEL

    print(f"Loading inception model: {args.inception_model}")
    device = 'cuda' if GPU_is_available() else 'cpu'

    state_dict = torch.load(args.inception_model, map_location=device)

    output_path = args.dir 
    output_path = mkdir_in_path(output_path, "evaluation_metrics")
    output_path = mkdir_in_path(output_path, "pkid")

    inception_cls = SpectrogramInception3(state_dict['fc.weight'].shape[0], aux_logits=False)
    inception_cls.load_state_dict(state_dict)
    
    
    inception_prepro = AudioPreprocessor(**DEFAULT_INCEPTION_PREPROCESSING_CONFIG).get_preprocessor()
    inception_score = []

    
    pbar = trange(int(np.ceil(n_samples/is_samples)), desc="MMD loop")
    mmd_distance = []
    for j in pbar:
        real_batch = true_files[j*is_samples:is_samples*(j+1)]
        fake_batch = fake_files[j*is_samples:is_samples*(j+1)]
        real_logits = []
        fake_logits = []
        
        for i in trange(int(np.ceil(len(real_batch)/batch_size)), desc='generating data'):
            real_input = map(inception_prepro, real_batch[i*batch_size:batch_size*(i+1)])
            real_input = torch.stack(list(real_input), dim=0)
            real_input = real_input[:, 0:1]
            real_input = F.interpolate(real_input, (299, 299))

            fake_input = map(inception_prepro, fake_batch[i*batch_size:batch_size*(i+1)])
            fake_input = torch.stack(list(fake_input), dim=0)
            fake_input = fake_input[:, 0:1]
            fake_input = F.interpolate(fake_input, (299, 299))

            real_logits.append(inception_cls(real_input).detach())
            fake_logits.append(inception_cls(fake_input).detach())


        real_logits = torch.cat(real_logits, dim=0)
        fake_logits = torch.cat(fake_logits, dim=0)
        
        mmd_distance.append(mmd(real_logits, fake_logits))
        mean_MMD = np.mean(mmd_distance)
        var_MMD = np.std(mmd_distance)
        pbar.set_description("PKID = {0:.4f} +- {1:.4f}".format(mean_MMD, var_MMD))
    output_file = f'{output_path}/PKID_{datetime.now().strftime("%y_%m_%d")}.txt'
    with open(output_file, 'w') as f:
        f.write(str(mean_MMD)+'\n')
        f.write(str(var_MMD))
        f.close()

