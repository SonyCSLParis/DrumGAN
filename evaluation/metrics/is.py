import torch
import numpy as np
import torch.nn.functional as F

from utils.utils import *
from datetime import datetime
from evaluation.train_inception_model import SpectrogramInception3
from data.preprocessing import AudioProcessor
from .inception_score import InceptionScore
from ..inception_models import DEFAULT_INSTRUMENT_INCEPTION_MODEL, DEFAULT_INCEPTION_PREPROCESSING_CONFIG
from data.audio_transforms import MelScale
from tqdm import trange


import ipdb

def test(parser):
    parser.add_argument('-c', dest="config", default="")
    parser.add_argument('--att', dest="attribute")
    # parser.add_argument('-b', '--batch-size', dest="batch_size", default=50)
    parser.add_argument('-N', dest='n_is', default=5000, 
        help="number of samples over which to compute IS")
    parser.add_argument('-n', '--name', dest='name', default='')
    args = parser.parse_args()

    assert os.path.exists(args.config), "Inception config not found"
    config = read_json(args.config)
    path = config["path"]

    gen_files = list(list_files_abs_path(args.fake_path, 'wav'))
    n_samples = len(gen_files)
    is_samples = min(n_samples, args.n_is)

    transform_config = config['transform_config']

    # HACK: this should go to audio_transforms.py 
    mel = MelScale(sample_rate=transform_config['sample_rate'],
                   fft_size=transform_config['fft_size'],
                   n_mel=transform_config.get('n_mel', 256),
                   rm_dc=True)

    print(f"Loading inception model: {config['path']}")
    device = 'cuda' if GPU_is_available() else 'cpu'

    state_dict = torch.load(config['path'], map_location=device)

    output_path = os.path.join(args.dir, "evaluation_metrics")
    checkexists_mkdir(output_path)
    

    inception_cls = SpectrogramInception3(state_dict['fc.weight'].shape[0], aux_logits=False)
    inception_cls.load_state_dict(state_dict)
    inception_cls = inception_cls.to(device)
    mel = mel.to(device)
    processor = AudioProcessor(**transform_config)
    inception_score = []
    print("Computing inception score on true data...\nYou can skip this with ctrl+c")
    try:
        pbar = trange(int(np.ceil(n_samples/is_samples)), desc="real data IS loop")
        proc_real = processor(gen_files)
        with torch.no_grad():
            for j in pbar:
                # processed_real = list(map(nsynth_prepro, gen_files[j*is_samples:is_samples*(j+1)]))
                is_maker = InceptionScore()
                is_data = torch.Tensor(proc_real[j*is_samples:is_samples*(j+1)])
                is_data = is_data[:, 0:1]
                
                for i in range(int(np.ceil(is_samples/args.batch_size))):
                    is_batch = is_data[i*args.batch_size:args.batch_size*(i+1)].to(device)
                    is_batch_data = mel(is_batch)
                    fake_data = F.interpolate(is_batch, (299, 299))
                    i_pred = inception_cls(fake_data.float()).detach().cpu()
                    # HACK:
                    i_pred = i_pred[:, -3:]
                    is_maker.updateWithMiniBatch(i_pred)
                    inception_score.append(is_maker.getScore())

                IS_mean = np.mean(inception_score)
                IS_std = np.std(inception_score)
                pbar.set_description("IIS = {0:.4f} +- {1:.4f}".format(IS_mean, IS_std/2.))
            output_file = f'{output_path}/IS_{str(n_samples)}_{args.name}_{datetime.now().strftime("%d-%m-%y_%H_%M")}.txt'
        with open(output_file, 'w') as f:
            f.write(str(IS_mean) + '\n')
            f.write(str(IS_std))
            f.close()
    except KeyboardInterrupt as k:
        print("Skipping true data inception score")

