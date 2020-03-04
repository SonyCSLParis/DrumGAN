import torch
import numpy as np
import torch.nn.functional as F
import subprocess

from .eval_utils import *
from ..utils.utils import GPU_is_available
from ..eval.train_inception_model import SpectrogramInception3
from ..datasets.datamanager import AudioDataManager
from ..metrics.inception_score import InceptionScore
from ..metrics.maximum_mean_discrepancy import mmd
from ..metrics.kernel_inception_distance import polynomial_mmd

from ..audio_processing import Compose
from tools import list_files_abs_path, checkexists_mkdir, mkdir_in_path
from tqdm import trange, tqdm
from random import shuffle
from audio.tools import saveAudioBatch

import ipdb

DEFAULT_INCEPTION_MODEL = \
    os.path.join(os.path.abspath(os.curdir), \
    "models/eval/inception_models/inteption_model_large_2019-11-18.pt")

def get_nsynth_IS_preprocessing(data_path, nsynth_att_path):
    return AudioDataManager(
                data_path=data_path,
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

    n_samples = 100
    part_size = 100
    batch_size = 5

    mean_is = -1
    mean_fad = -1
    mean_mmd = -1
    mean_pmmd = -1
    is_real = -1
    fad_real = -1
    mmd_real = -1
    pmmd_real = -1
    is_list_real = []
    is_list_fake = []
    fad_list_fake = []
    fad_list_true = []
    mmd_list_fake = []
    mmd_list_true = []
    pmmd_list_true = []
    pmmd_list_fake = []

    args = parser.parse_args()
    compute_bounds = args.compute_bounds
    compute_is = args.compute_is
    compute_fad = args.compute_fad
    compute_mmd = args.compute_mmd
    compute_pmmd = args.compute_pmmd

    kwargs = vars(args)
    data_path = kwargs.get('data_path')
    device = 'cuda' if GPU_is_available() else 'cpu'
    use_gpu = GPU_is_available()
    # load model
    model, config, model_name, path = \
        load_model_checkp(**kwargs, useGPU=use_gpu)
    # create output folders
    output_path = os.path.join(path, "output")
    checkexists_mkdir(output_path)
    output_path = mkdir_in_path(output_path, "EVAL")
    output_path = mkdir_in_path(output_path, model_name)
    
    # overwrite some data configruations
    data_config = config["dataConfig"]
    # overwrite data  and metadata paths
    data_config["data_path"] = os.path.join(data_path, 'audio')
    data_config["loaderConfig"]["att_dict_path"] = \
        os.path.join(data_path, 'examples.json')
    # overwrite data_config from model with /tmp/ output path
    data_config["output_path"] = "/tmp"
    data_config["loaderConfig"]["size"] = n_samples
    # set preprocess to False for reducing computation
    data_config["loaderConfig"]['preprocess'] = False
    # load data and get post-processor
    real_data_loader = AudioDataManager(**data_config, preprocess=False)
    model_postpro = real_data_loader.get_post_processor()
    real_files = real_data_loader.loader.data
    shuffle(real_files)
    real_files = real_files[:n_samples]
    n_samples = min(len(real_files), n_samples)

    # load inception model
    print("Loading inception model...\n")
    is_prepro = get_nsynth_IS_preprocessing(
        os.path.join(data_path, 'audio'),
        os.path.join(data_path, 'examples.json'))
    # get number of attributes
    att = kwargs.get('att_name', 'pitch')
    if att == 'pitch':
        n_classes = len(real_data_loader.loader.att_count[att])
    elif att == 'instrument':
        n_classes = len(real_data_loader.loader.instrument_labels)
    inception_model_path = kwargs.pop('imodel', DEFAULT_INCEPTION_MODEL)
    assert att in inception_model_path, "Att class label not in inception model name!!!!"
    state_dict = torch.load(inception_model_path, map_location=device)
    inception_cls = SpectrogramInception3(n_classes, aux_logits=False)
    inception_cls.load_state_dict(state_dict)

    real_data_prepro = real_data_loader.pre_pipeline[:3]
    if False:
        real_data_prepro = real_data_loader.pre_pipeline
    tmp_output_real = mkdir_in_path('/tmp', 'FAD_basic_processed_true' + model_name)
    tmp_output_fake = mkdir_in_path('/tmp', 'FAD_basic_processed_' + model_name)

    n_parts = int(np.ceil(n_samples/part_size))
    main_bar = trange(n_parts, desc='main evaluation loop')
    print("!!!!!!!STARTING EVALUATION!!!!!!!!!\n")
    for a in main_bar:
        part_files = real_files[a*part_size:part_size*(a+1)]
        n_batch = int(np.ceil(len(part_files)/batch_size))

        is_maker_real = InceptionScore(inception_cls)
        is_maker_fake = InceptionScore(inception_cls)
        
        real_logits = []
        fake_logits = []
        real_logits1 = []
        real_logits2 = []

        for b in trange(n_batch, desc='mini-batch loop'):
            batch_files = part_files[b*batch_size:batch_size*(b+1)]
            input_z, labels = model.buildNoiseData(batch_size, None, skipAtts=True)
            y = model.test(input_z.to(device), getAvG=True, toCPU=not use_gpu).detach().cpu()
            y = list(tqdm(map(model_postpro, y), desc='post-processing generated data'))
            y = list(map(lambda x: np.array(x).astype(float), y))

            if compute_is or compute_mmd or compute_pmmd:

                inception_x = map(Compose(is_prepro), batch_files)
                inception_x = list(tqdm(inception_x, desc='1.1) IS Processing'))
                inception_x = torch.stack(inception_x, dim=0)
                inception_x = inception_x[:, 0:1] # get magnitude spec
                inception_x = F.interpolate(inception_x, (299, 299))
                
                inception_y = tqdm(map(Compose(is_prepro[2:]), y), desc='1.3) IS preprocessing')
                inception_y = torch.stack(list(inception_y), dim=0)
                inception_y = inception_y[:, 0:1]
                inception_y = F.interpolate(inception_y, (299, 299))
            
            if compute_is:
                if compute_bounds:
                    is_maker_real.updateWithMiniBatch(inception_x)
                # inception score on fake data
                is_maker_fake.updateWithMiniBatch(inception_y)

            if compute_fad:
                saveAudioBatch(y,
                               tmp_output_fake, 
                               overwrite=True, 
                               basename=f'gen_audio_{b}')
            if compute_mmd or compute_pmmd:
                real_logits.append(inception_cls(inception_x).detach())
                fake_logits.append(inception_cls(inception_y).detach())

                if compute_bounds:
                    real_logits1.append(inception_cls(inception_x[:int(len(inception_x)/2)]).detach())
                    real_logits2.append(inception_cls(inception_x[int(len(inception_x)/2):]).detach())

        if compute_is:

            print("Computing IS")
            if compute_bounds:
                is_list_real.append(is_maker_real.getScore())
                is_real = np.mean(is_list_real)
                is_std_real = np.std(is_list_real)
                main_bar.set_description("True IS = {0:.4f} +- {1:.4f}".format(is_real, is_std_real/2.))
                            
                output_file = f'{output_path}/True_IS_{att}_{model_name}.txt'
                with open(output_file, 'w') as f:
                    f.write(str(is_real) + '\n')
                    f.write(str(is_real))
                    f.close()

            is_list_fake.append(is_maker_fake.getScore())
            mean_is = np.mean(is_list_fake)
            is_std_fake = np.std(is_list_fake)

            output_file = f'{output_path}/Fake_IS_{att}_{model_name}.txt'
            with open(output_file, 'w') as f:
                f.write(str(mean_is) + '\n')
                f.write(str(is_std_fake))
                f.close()

        if compute_fad:
            print("computing FAD")
            real_paths_csv = f"{output_path}/real_audio.cvs"
            with open(real_paths_csv, "w") as f:
                for file_path in part_files:
                    f.write(file_path + '\n')

            fake_files = list(list_files_abs_path(tmp_output_fake, 'wav'))
            fake_paths_csv = f"{output_path}/fake_audio.cvs"
            with open(fake_paths_csv, "w") as f:
                for file_path in fake_files:
                    f.write(file_path + '\n')

            fad_list_fake.append(float(subprocess.check_output(["sh",
                                "shell_scripts/run_fad_eval.sh",
                                "--real-path="+real_paths_csv,
                                "--fake-path="+fake_paths_csv,
                                "--model-root-path="+output_path]).decode()[-10:-1]))
            mean_fad = np.mean(fad_list_fake)
            with open(f"{output_path}/fad_real-fake_{str(n_samples)}.txt", "w") as f:
                f.write(str(mean_fad))
                f.close()

            if compute_bounds:
                real_files1 = batch_files[:int(len(batch_files) / 2)]
                real_files2 = batch_files[int(len(batch_files) / 2):]

                real_paths_csv = f"{output_path}/real_audio.cvs"
                with open(real_paths_csv, "w") as f:
                    for file_path in real_files1:
                        f.write(file_path + '\n')

                fake_paths_csv = f"{output_path}/fake_audio.cvs"
                with open(fake_paths_csv, "w") as f:
                    for file_path in real_files2:
                        f.write(file_path + '\n')

                fad_list_true.append(float(subprocess.check_output(["sh",
                                    "shell_scripts/run_fad_eval.sh",
                                    "--real-path="+real_paths_csv,
                                    "--fake-path="+fake_paths_csv,
                                    "--model-root-path="+output_path]).decode()[-10:-1]))
                fad_real = np.mean(fad_list_true)
                with open(f"{output_path}/fad_real-real_{str(n_samples)}.txt", "w") as f:
                    f.write(str(fad_real))
                    f.close()

                print("FAD_real_50k={0:.4f}".format(fad_real))
        if compute_pmmd:
            real_logits = torch.cat(real_logits, dim=0)
            fake_logits = torch.cat(fake_logits, dim=0)
            pmmd_list_fake.append(polynomial_mmd(real_logits, fake_logits))
            mean_pmmd = np.mean(pmmd_list_fake)
            pmmd_fake_std = np.std(pmmd_list_fake)
            output_file = f'{output_path}/pkernel_MMD_50k_fake-real_{model_name}.txt'
            with open(output_file, 'w') as f:
                f.write(str(mean_pmmd)+'\n')
                f.write(str(pmmd_fake_std))
                f.close()
        if compute_mmd:
            print("Computing MMD")
            real_logits = torch.cat(real_logits, dim=0)
            fake_logits = torch.cat(fake_logits, dim=0)
            mmd_list_fake.append(mmd(real_logits, fake_logits))
            mean_mmd = np.mean(mmd_list_fake)
            mmd_fake_std = np.std(mmd_list_fake)
            output_file = f'{output_path}/MMD_50k_fake-real_{model_name}.txt'
            with open(output_file, 'w') as f:
                f.write(str(mean_mmd)+'\n')
                f.write(str(mmd_fake_std))
                f.close()

            if compute_bounds:
                real_logits1 = torch.cat(real_logits1, dim=0)
                real_logits2 = torch.cat(real_logits2, dim=0)
                mmd_list_true.append(mmd(real_logits1, real_logits2))
                mmd_real = np.mean(mmd_list_true)
                mmd_real_std = np.std(mmd_list_true) 
                output_file = f'{output_path}/MMD_50k_real-real_{model_name}.txt'
                with open(output_file, 'w') as f:
                    f.write(str(mmd_real)+'\n')
                    f.write(str(mmd_real_std))
                    f.close()


    print("ISreal={0:.4f}\n\
           ISfake={1:.4f}\n\
           FADreal={2:.4f}\n\
           FADfake={3:.4f}\n\
           MMDreal={4:.4f}\n\
           MMDfake={5:.4f}\n\
           PMMDfake={6:.4f}, ".format(
                               is_real, 
                               mean_is, 
                               fad_real, 
                               mean_fad, 
                               mmd_real,
                               mean_mmd,
                               mean_pmmd))