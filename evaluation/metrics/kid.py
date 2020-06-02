import torch
import numpy as np
import torch.nn.functional as F

from utils.utils import *
from datetime import datetime
from evaluation.train_inception_model import SpectrogramInception3
from .maximum_mean_discrepancy import mmd

from data.preprocessing import AudioProcessor
from data.audio_transforms import MelScale
from ..inception_models import DEFAULT_PITCH_INCEPTION_MODEL, DEFAULT_INCEPTION_PREPROCESSING_CONFIG
from tqdm import trange


def test(parser, visualisation=None):
    parser.add_argument('-c', dest="config", default="")
    parser.add_argument('--att', dest="attribute")
    parser.add_argument('-N', dest='n_is', default=5000, 
        help="number of samples over which to compute IS")
    args = parser.parse_args()

    assert os.path.exists(args.config), "Inception config not found"
    config = read_json(args.config)
    path = config["path"]

    true_files = list(list_files_abs_path(args.true_path, 'wav'))
    fake_files = list(list_files_abs_path(args.fake_path, 'wav'))
    n_samples = len(fake_files)
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

    
    pbar = trange(int(np.ceil(n_samples/is_samples)), desc="MMD loop")
    mmd_distance = []
    with torch.no_grad():
        proc_true = processor(true_files)
        proc_fake = processor(fake_files)
        for j in pbar:
            real_batch = torch.Tensor(proc_true[j*is_samples:is_samples*(j+1)])
            fake_batch = torch.Tensor(proc_fake[j*is_samples:is_samples*(j+1)])
            real_logits = []
            fake_logits = []
            
            for i in trange(int(np.ceil(len(real_batch)/args.batch_size)), desc='inception loop'):
                real_input = real_batch[i*args.batch_size:args.batch_size*(i+1)].to(device)
                real_input = mel(real_input)
                # real_input = torch.stack(list(real_input), dim=0)
                real_input = real_input[:, 0:1]
                real_input = F.interpolate(real_input, (299, 299))

                fake_input = fake_batch[i*args.batch_size:args.batch_size*(i+1)].to(device)
                fake_input = mel(fake_input)
                # fake_input = torch.stack(list(fake_input), dim=0)
                fake_input = fake_input[:, 0:1]
                fake_input = F.interpolate(fake_input, (299, 299))

                real_logits.append(inception_cls(real_input).detach().cpu())
                fake_logits.append(inception_cls(fake_input).detach().cpu())


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

