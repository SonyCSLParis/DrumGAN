import torch
import numpy as np
import torch.nn.functional as F

from .eval_utils import *
from ..utils.utils import GPU_is_available
from ..eval.train_inception_model import SpectrogramInception3
from ..datasets.datamanager import AudioDataManager
from ..metrics.inception_score import InceptionScore
from ..audio_processing import Compose
from tools import list_files_abs_path, checkexists_mkdir, mkdir_in_path
from tqdm import trange

import ipdb

DEFAULT_INCEPTION_MODEL = \
    os.path.join(os.path.abspath(os.curdir), \
    "models/eval/inception_models/inteption_model_large_2019-11-18.pt")

def get_nsynth_IS_preprocessing(nsynth_path, nsynth_att_path):
    return AudioDataManager(
                data_path=nsynth_path,
                output_path="/tmp/",
                dbname='nsynth',
                sample_rate=16000,
                audio_len=16000,
                data_type='audio',
                transform='specgrams',
                db_size=1,
                labels=[],
                transformConfig=dict(
                    n_frames=64,
                    n_bins=128,
                    fade_out=True,
                    fft_size=2048,
                    win_size=1024,
                    hop_size=256,
                    n_mel=256
                ),
                load_metadata=True,
                loaderConfig=dict(
                    size=1,
                    instrument_labels=['all'],
                    filter_keys=['acoustic'],
                    attribute_list=["instrument_family"],
                    att_dict_path=nsynth_att_path,
                    )).pre_pipeline

def test(parser, visualisation=None):
    args = parser.parse_args()

    n_samples = 100
    is_samples = 100
    batch_size = 100
    kwargs = vars(args)
    nsynth_path = kwargs.get('data_path')
    att = kwargs.get('att_name', 'pitch')

    gen_files = list(list_files_abs_path(args.fake_path, 'wav'))

    inception_model_path = kwargs.pop('imodel', DEFAULT_INCEPTION_MODEL)
    assert att in inception_model_path, "Att class label not in inception model name!!!!"

    print(f"Loading inception model: {inception_model_path}")
    device = 'cuda' if GPU_is_available() else 'cpu'

    state_dict = torch.load(inception_model_path, \
        map_location=device)

    output_path = os.path.join(args.dir, "output")
    checkexists_mkdir(output_path)
    

    inception_cls = SpectrogramInception3(state_dict['fc.weight'].shape[0], aux_logits=False)
    inception_cls.load_state_dict(state_dict)
    
    
    nsynth_prepro = get_nsynth_IS_preprocessing(os.path.join(nsynth_path, 'audio'),
                                                os.path.join(nsynth_path, 'examples.json'))
    inception_score = []
    print("Computing inception score on true data...\nYou can skip this with ctrl+c")
    try:
        pbar = trange(int(n_samples/is_samples), desc="real data IS loop")
        for j in pbar:
            processed_real = list(map(Compose(nsynth_prepro), gen_files[j*is_samples:is_samples*(j+1)]))
            is_maker = InceptionScore(inception_cls)
            is_data = torch.stack(processed_real, dim=0)
            is_data = is_data[:, 0:1]
            
            for i in range(int(np.ceil(is_samples/batch_size))):

                fake_data = F.interpolate(is_data[i*batch_size:batch_size*(i+1)], (299, 299))
                is_maker.updateWithMiniBatch(fake_data)
                inception_score.append(is_maker.getScore())

            IS_mean = np.mean(inception_score)
            IS_std = np.std(inception_score)
            pbar.set_description("IS_50k = {0:.4f} +- {1:.4f}".format(IS_mean, IS_std/2.))
        output_file = f'{output_path}/IS_50k_{att}_real-real_{model_name}.txt'
        with open(output_file, 'w') as f:
            f.write(str(IS_mean) + '\n')
            f.write(str(IS_std))
            f.close()
    except KeyboardInterrupt as k:
        print("Skipping true data inception score")

