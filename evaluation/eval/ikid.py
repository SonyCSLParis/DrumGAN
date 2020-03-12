import torch
import numpy as np
import torch.nn.functional as F

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
    n_samples = 1000
    is_samples = 100
    batch_size = 10
    kwargs = vars(parser.parse_args())
    nsynth_path = kwargs.get('data_path')

    model, config, model_name, path = load_model_checkp(**kwargs, useGPU=GPU_is_available())
    inception_model_path = kwargs.pop('inception_model', DEFAULT_INCEPTION_MODEL)
    state_dict = torch.load(inception_model_path, \
        map_location='cuda' if GPU_is_available() else 'cpu')

    output_path = os.path.join(path, "output")
    checkexists_mkdir(output_path)
    output_path = mkdir_in_path(output_path, "MMD")

    output_path = mkdir_in_path(output_path, model_name)
    
    inception_score = []
    inception_cls = SpectrogramInception3(11, aux_logits=False)
    inception_cls.load_state_dict(state_dict)
    
    model_postpro = get_dummy_nsynth_loader(config, nsynth_path).get_post_processor()
    nsynth_prepro = get_nsynth_IS_preprocessing(os.path.join(nsynth_path, 'audio'),
                                                os.path.join(nsynth_path, 'examples.json'))

    print("Computing MMD distance on true data...\nYou can skip this with ctrl+c")
    try:
        real_files = list(tqdm(list_files_abs_path(os.path.join(nsynth_path, 'audio'), 'wav'), desc='Reading files...'))
        shuffle(real_files)
        real_files = real_files[:n_samples]

        rfiles = real_files[:int(len(real_files)/2)]
        ffiles = real_files[int(len(real_files)/2):]
        processed_real = []
        pbar = trange(int(len(rfiles)/is_samples), desc="real data MMD loop")
        mmd_distance = []
        for j in pbar:
            real_input = list(tqdm(map(Compose(nsynth_prepro), rfiles[j*is_samples:is_samples*(j+1)]), desc='Processing data...'))
            fake_input = list(tqdm(map(Compose(nsynth_prepro), ffiles[j*is_samples:is_samples*(j+1)]), desc='Processing data...'))

            rbatch = torch.stack(real_input, dim=0)[:, 0:1]
            fbatch = torch.stack(fake_input, dim=0)[:, 0:1]

            processed_real.append(rbatch)
            processed_real.append(fbatch)

            real_logits = []
            fake_logits = []
            for i in trange(int(np.ceil(is_samples/batch_size)), desc='Computing inception features...'):
                real_data = F.interpolate(rbatch[i*batch_size:batch_size*(i+1)], (299, 299))
                fake_data = F.interpolate(fbatch[i*batch_size:batch_size*(i+1)], (299, 299))
                

                real_logits.append(inception_cls(real_data).detach())
                fake_logits.append(inception_cls(fake_data).detach())

            real_logits = torch.cat(real_logits, dim=0)
            fake_logits = torch.cat(fake_logits, dim=0)
            mmd_distance.append(mmd(real_logits, fake_logits))
            mean_MMD = np.mean(mmd_distance)
            var_MMD = np.std(mmd_distance)

            pbar.set_description("MMD_50k= {0:.4f} +- {1:.4f}".format(mean_MMD, var_MMD))
        output_file = f'{output_path}/MMD_50k_real-real_{model_name}.txt'
        with open(output_file, 'w') as f:
            f.write(str(mean_MMD)+'\n')
            f.write(str(var_MMD))
            f.close()
    except KeyboardInterrupt as k:
        print("Skipping true data inception score")

    print("Computing inception score on generated data...")
    processed_real = torch.cat(processed_real, dim=0)
    
    pbar2 = trange(int(len(processed_real)/is_samples), desc="fake data MMD loop")
    mmd_distance = []
    for j in pbar2:
        rbatch = processed_real[j*is_samples:is_samples*(j+1)]

        real_logits = []
        fake_logits = []
        
        for i in trange(int(np.ceil(is_samples/batch_size)), desc='generating data'):
            input_z, labels = model.buildNoiseData(batch_size, None, skipAtts=True)
            sample_block = model.test(input_z, getAvG=True)
            gen_raw = map(model_postpro, sample_block)
            gen_raw = map(lambda x: np.array(x), gen_raw)
            fake_input = map(Compose(nsynth_prepro[2:]), gen_raw)

            fake_input = torch.stack(list(fake_input), dim=0)
            fake_input = fake_input[:, 0:1]
            
            real_data = F.interpolate(rbatch[i*batch_size:batch_size*(i+1)], (299, 299))
            fake_data = F.interpolate(fake_input, (299, 299))


            real_logits.append(inception_cls(real_data).detach())
            fake_logits.append(inception_cls(fake_data).detach())


        real_logits = torch.cat(real_logits, dim=0)
        fake_logits = torch.cat(fake_logits, dim=0)
        
        mmd_distance.append(mmd(real_logits, fake_logits))
        mean_MMD = np.mean(mmd_distance)
        var_MMD = np.std(mmd_distance)
        pbar2.set_description("MMD_50k= {0:.4f} +- {1:.4f}".format(mean_MMD, var_MMD))
    output_file = f'{output_path}/MMD_50k_fake-real_{model_name}.txt'
    with open(output_file, 'w') as f:
        f.write(str(mean_MMD)+'\n')
        f.write(str(var_MMD))
        f.close()

