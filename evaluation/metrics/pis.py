import torch
import numpy as np
import torch.nn.functional as F

from utils.utils import *
from datetime import datetime
from evaluation.train_inception_model import SpectrogramInception3
from data.preprocessing import AudioPreprocessor
from .inception_score import InceptionScore
from ..inception_models import DEFAULT_PITCH_INCEPTION_MODEL, DEFAULT_INCEPTION_PREPROCESSING_CONFIG
from data.audio_transforms import Compose
from tqdm import trange



def test(parser):
    args = parser.parse_args()

    kwargs = vars(args)
    nsynth_path = kwargs.get('data_path')
    att = kwargs.get('att_name', 'pitch')
    batch_size = kwargs.get('batch_size', 50)
    is_samples = kwargs.get('is_samples', 5000)

    gen_files = list(list_files_abs_path(args.fake_path, 'wav'))
    n_samples = len(gen_files)
    is_samples = min(n_samples, is_samples)

    if args.inception_model == None:
        args.inception_model = DEFAULT_PITCH_INCEPTION_MODEL

    print(f"Loading inception model: {args.inception_model}")
    device = 'cuda' if GPU_is_available() else 'cpu'

    state_dict = torch.load(args.inception_model, map_location=device)

    output_path = os.path.join(args.dir, "evaluation_metrics")
    checkexists_mkdir(output_path)
    

    inception_cls = SpectrogramInception3(state_dict['fc.weight'].shape[0], aux_logits=False)
    inception_cls.load_state_dict(state_dict)
    
    
    nsynth_prepro = AudioPreprocessor(**DEFAULT_INCEPTION_PREPROCESSING_CONFIG).get_preprocessor()
    inception_score = []
    print("Computing inception score on true data...\nYou can skip this with ctrl+c")
    try:
        pbar = trange(int(n_samples/is_samples), desc="real data IS loop")
        for j in pbar:
            processed_real = list(map(nsynth_prepro, gen_files[j*is_samples:is_samples*(j+1)]))
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
        output_file = f'{output_path}/PIS_{str(n_samples)}_{datetime.now().strftime("%d-%m-%y")}.txt'
        with open(output_file, 'w') as f:
            f.write(str(IS_mean) + '\n')
            f.write(str(IS_std))
            f.close()
    except KeyboardInterrupt as k:
        print("Skipping true data inception score")

