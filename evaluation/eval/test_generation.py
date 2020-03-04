import os
import json
# import torchvision
import torch
import subprocess
from datetime import datetime
from .eval_utils import *
from os.path import dirname, realpath, join
from ..metrics.inception_score import InceptionScore
from ..utils.utils import printProgressBar, read_json
from ..utils.utils import getVal, loadmodule, getLastCheckPoint, \
    parse_state_name, getNameAndPackage, saveScore
from ..networks.constant_net import FeatureTransform

from .train_inception_model import SpectrogramInception3
from ..datasets.datamanager import AudioDataManager

from ..metrics.frechet_inception_distance import FrechetInceptionDistance
import ipdb
from torch.utils.data import DataLoader
import numpy as np

from audio.tools import saveAudioBatch

import matplotlib.pyplot as plt 
from ..utils.rainbowgram.wave_rain import wave2rain
from ..utils.rainbowgram.rain2graph import rain2graph

from tools import mkdir_in_path, get_filename, checkexists_mkdir

from .generation_tests import *
import traceback

from .eval_utils import *

NSYNTH_PATH = "/Users/javier/Developer/datasets/nsynth-train/"
assert os.path.exists(NSYNTH_PATH), f"Cannot find Nsynth at {NSYNTH_PATH}"


def test(parser, visualisation=None):
    kwargs = vars(parser.parse_args())
    test_list = kwargs.pop("test_list", [0, 1, 2, 3, 4, 5, 6, 7])

    model, config, model_name, path = load_model_checkp(**kwargs)
    latentDim = model.config.categoryVectorDim_G

    # We load a dummy data loader for post-processing
    dummy_loader = get_dummy_nsynth_loader(config, nsynth_path=NSYNTH_PATH)
    postprocess = dummy_loader.get_post_processor()

    # Create output evaluation dir
    output_dir = mkdir_in_path(path, f"tests")
    output_dir_name = f"test_{model_name}"
    output_dir = mkdir_in_path(output_dir, output_dir_name)
    output_dir = mkdir_in_path(output_dir, datetime.now().strftime('%Y-%m-%d %H:%M'))
    # Create evaluation manager
    eval_manager = StyleGEvaluationManager(model, n_gen=100)

    ################ TEST RANDOM GENERATION ##################
    if 0 in test_list:
        print("TEST RANDOM GENNERATION")
        output_path = mkdir_in_path(output_dir, f"rand_gen")
        gen_batch = eval_manager.test_random_generation()
        audio_out = map(postprocess, gen_batch)

        saveAudioBatch(audio_out,
                       path=output_path,
                       basename='test_rand_gen', 
                       sr=config["dataConfig"]["sample_rate"])
        print("FINISHED!\n")
    ###########################################################

    ################ TEST FRECHET AUDIO DISTANCE ##################
    if 1 in test_list:
        # put test into generation_tests.py
        print("TEST FRECHET AUDIO DISTANCE")
        output_path = mkdir_in_path(output_dir, "FAD")
        fad_real_path = os.path.join(output_path, "real")
        fad_fake_path = os.path.join(output_path, "fake")
        b_size = 10
        if not checkexists_mkdir(fad_fake_path):
            input_z = model.buildNoiseData(len(dummy_loader.loader.data), skipAtts=True)[0]
            if len(input_z) > b_size:
                gen_batch = []
                for i in range(int(np.floor(len(input_z) / b_size))):
                    gen_batch += [model.test(input_z[i*b_size:b_size*(i + 1)], toCPU=True, getAvG=False)]

                gen_batch = torch.cat(gen_batch, dim=0)
            else:
                gen_batch.append(model.test(input_z, toCPU=True, getAvG=False))
            fake_audio_out = map(postprocess, gen_batch)
            saveAudioBatch(fake_audio_out,
                           path=fad_fake_path,
                           basename='fake', 
                           sr=config["dataConfig"]["sample_rate"])

        if not checkexists_mkdir(fad_real_path):
            loader = iter(torch.utils.data.DataLoader(dummy_loader.loader,
                                                 batch_size=len(dummy_loader.loader),
                                                    shuffle=True))
            data, _ = loader.next()

            # audio_out = map(torch.from_numpy, data)   
            audio_out = map(postprocess, data)

            saveAudioBatch(audio_out,
                           path=fad_real_path,
                           basename='real', 
                           sr=config["dataConfig"]["sample_rate"])

        subprocess.call(["sh",
                        "shell_scripts/run_fad.sh",
                        "--real-path="+fad_real_path,
                        "--fake-path="+fad_fake_path,
                        "--model-root-path="+output_path])
    ###########################################################

    ################ TEST SINGLE PITCH - RAND Z ###############
    if 2 in test_list:
        try:
            print("TEST SINGLE PITCH - RAND Z")
            gen_batch = eval_manager.test_single_pitch_random_z()
            output_path = mkdir_in_path(output_dir, f"one_pitch_rand_z")
            audio_out = map(postprocess, gen_batch)
            saveAudioBatch(audio_out,
                           path=output_path,
                           basename='test_single_p_rand_z', 
                           sr=config["dataConfig"]["sample_rate"])
            print("FINISHED!\n")
        except Exception as e:
            print("ERROR in TEST SINGLE PITCH - RAND Z")
            print(e)
    ###########################################################


    ################ TEST SINGLE PITCH - Z INTERPOLATION ######
    if 3 in test_list:
        print("TEST SINGLE PITCH - Z INTERPOLATION")
        try:
            gen_batch = eval_manager.test_single_pitch_latent_interpolation()

            output_path = mkdir_in_path(output_dir, f"one_pitch_z_interp")
            
            audio_out = map(postprocess, gen_batch)
            saveAudioBatch(audio_out,
                           path=output_path,
                           basename='test_z_interp', 
                           sr=config["dataConfig"]["sample_rate"])
            print("FINISHED\n")
        except Exception as e:
            print(f"Error: {e}")
            print(traceback.print_tb(e.__traceback__))
    ###########################################################

        print("SPHERICAL INTERPOLATION")
        try: 
            gen_batch = eval_manager.test_single_pitch_sph_latent_interpolation()
            output_path = mkdir_in_path(output_dir, f"spherical_z_interp")
            audio_out = map(postprocess, gen_batch)
            saveAudioBatch(audio_out,
                           path=output_path,
                           basename='spherical_interp', 
                           sr=config["dataConfig"]["sample_rate"])
            print("FINISHED\n")
        except Exception as e:
            print(f"Error: {e}")
            print(traceback.print_tb(e.__traceback__))
    ###########################################################
        print("SURFACE INTERPOLATION")
        try:
            gen_batch = eval_manager.test_single_pitch_sph_surface_interpolation()
            output_path = mkdir_in_path(output_dir, f"sph_surface_z_interp")
            audio_out = map(postprocess, gen_batch)
            saveAudioBatch(audio_out,
                           path=output_path,
                           basename='sph_surface_interp', 
                           sr=config["dataConfig"]["sample_rate"])
            print("FINISHED\n")
        except Exception as e:
            print(f"Error: {e}")
            print(traceback.print_tb(e.__traceback__))

    ############### TEST SINGLE Z - PITCH SWEEP ###############
    if 4 in test_list:
        try:
            print("TEST PITCH SWEEP")
            gen_batch = eval_manager.test_single_z_pitch_sweep()
            output_path = mkdir_in_path(output_dir, f"one_z_pitch_sweep")
            audio_out = map(postprocess, gen_batch)
            saveAudioBatch(audio_out,
                           path=output_path,
                           basename='test_pitch_sweep', 
                           sr=config["dataConfig"]["sample_rate"])
            print("FINISHED!\n")
        except Exception as e:
            print(f"Error: {e}")
    ###########################################################
    

    ############### TEST FULL ATTRIBUTE SWEEP ###############
    if 5 in test_list:
        print("TEST FULL ATT SWEEP")
        try:
            gen_batch = eval_manager.test_full_attribute_sweep()
            output_path = mkdir_in_path(output_dir, f"full_att_sweep")
            audio_out = map(postprocess, gen_batch)
            saveAudioBatch(audio_out,
                           path=output_path,
                           basename='test_rand_gen', 
                           sr=config["dataConfig"]["sample_rate"])
            print("FINISHED!\n")
        except Exception as e:
            print(f"Error: {e}")
    ###########################################################
    
    ############### TEST MULTI-SCALE STYLE GEN ###############
    if 6 in test_list:
        print("TEST MULTI-SCALE STYLE GEN")
        try:
            gen_batch = eval_manager.test_multi_scale_gen()
            output_path = mkdir_in_path(output_dir, f"multi-scale_gen2")
            audio_out = map(postprocess, gen_batch)

            saveAudioBatch(audio_out,
                           path=output_path,
                           basename='test_ms_gen', 
                           sr=config["dataConfig"]["sample_rate"])
            print("FINISHED!\n")
        except Exception as e:
            print(f"Error: {e}")
    ###########################################################


    ############### TEST MEAN STYLE GEN ###############
    if 7 in test_list:
        print("TEST MEAN STYLE GEN")
        try:
            gen_batch = eval_manager.test_mean_weight_style()
            output_path = mkdir_in_path(output_dir, f"mean_style8")
            audio_out = map(postprocess, gen_batch)
            saveAudioBatch(audio_out,
                           path=output_path,
                           basename='test_mean_style_gen', 
                           sr=config["dataConfig"]["sample_rate"])
            print("FINISHED!\n")
        except Exception as e:
            print(f"Error: {e}")
    ###########################################################


    ############### TEST NOISE INJECTION ###############
    # print("NOISE INJECTION")
    # gen_batch = eval_manager.test_noise_injection()
    # output_path = mkdir_in_path(output_dir, f"full_att_sweep")
    # audio_out = map(postprocess, gen_batch)
    # saveAudioBatch(audio_out,
    #                path=output_path,
    #                basename='test_noise_inj', 
    #                sr=config["dataConfig"]["sample_rate"])
    # print("FINISHED!\n")
    ###########################################################
    

    # print("Extracting RAINBOWGRAMS ")
    # # RAINBOWGRAMS
    # output_path = mkdir_in_path(roo_out, "true_data")
    # for i, d in enumerate(data_loader.data):
    #     extract_save_rainbowgram(d, path=output_path, name=f'true_data_{i}')

