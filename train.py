print("Running training script...")
import ipdb
import os
import sys
import importlib
import argparse
import git

import time

from utils.utils import getVal, getLastCheckPoint, loadmodule, GPU_is_available
from utils.config import getConfigOverrideFromParser, updateParserWithConfig

from tools import checkexists_mkdir, mkdir_in_path
from pg_gan.progressive_gan_trainer import ProgressiveGANTrainer
from data.data_manager import AudioDataManager
from numpy import random
import torch
import torch.backends.cudnn as cudnn

import json



def init_seed(rand_seed=True):
    if not rand_seed:
        seed = random.randint(0, 9999)
    else:
        seed = 0

    random.seed(seed)
    torch.manual_seed(seed)

    if GPU_is_available():
        torch.cuda.manual_seed_all(baseArgs.seed)
    print("Random Seed: ", baseArgs.seed)
    print()

def load_config_file(configPath):
    if configPath is None:
        raise ValueError("You need to input a configuratrion file")
    with open(configPath, 'rb') as file:
        return json.load(file)

def save_config_file(configFile, outputPath):
    with open(outputPath, 'w') as file:

        return json.dump(configFile, file, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Testing script')
    parser.add_argument('model_name', type=str, default='PGAN',
                         help='Name of the model to launch, available models are\
                         PGAN and PPGAN. To get all possible option for a model\
                         please run train.py $MODEL_NAME -overrides')
   
    parser.add_argument('data_type', type=str, default=None, nargs='?',
                         help='Type of the data fed into the model, available types are\
                         IMAGE, AUDIO and SPECTRUM. To get all possible option for a model\
                         please run train.py $MODEL_NAME -overrides')
    
    parser.add_argument('--no_vis', help=' Disable all visualizations',
                        action='store_true')
    parser.add_argument('--np_vis', help=' Replace visdom by a numpy based \
                        visualizer (SLURM)',
                        action='store_true')
    parser.add_argument('--restart', help=' If a checkpoint is detected, do \
                                           not try to load it',
                        action='store_true')
    parser.add_argument('-n', '--name', help="Model's name",
                        type=str, dest="name", default="default")
    parser.add_argument('-d', '--dir', help='Output directory',
                        type=str, dest="dir", default='output_networks')
    parser.add_argument('-c', '--config', help="Path to configuration file",
                        type=str, dest="configPath")
    parser.add_argument('-s', '--save_iter', help="If it applies, frequence at\
                        which a checkpoint should be saved. In the case of a\
                        evaluation test, iteration to work on.",
                        type=int, dest="saveIter", default=16000)
    parser.add_argument('-e', '--eval_iter', help="If it applies, frequence at\
                        which a checkpoint should be saved",
                        type=int, dest="evalIter", default=100)
    parser.add_argument('-S', '--Scale_iter', help="If it applies, scale to work\
                        on")
    parser.add_argument('-v', '--partitionValue', help="Partition's value",
                        type=str, dest="partition_value")
    
    parser.add_argument('--render_audio', help="Render audio?",
                        action='store_true')

    # parser.add_argument('--data_type', type=str, default='audio', help="Data is Image, audio or spectrogram")
    parser.add_argument('--seed', action='store_true', help="Partition's value")
    parser.add_argument('--n_samples', type=int, default=10, help="Partition's value")
    
    # Parse command line args
    baseArgs, unknown = parser.parse_known_args()

     # Retrieve the model we want to launch
    print(f"Loading traines for {baseArgs.model_name}")

    # Initialize random seed
    init_seed(baseArgs.seed)
    
    # Build the output directory if necessary
    checkexists_mkdir(baseArgs.dir)

    # Add overrides to the parser: changes to the model configuration can be
    # done via the command line
    parser = updateParserWithConfig(parser, ProgressiveGANTrainer._defaultConfig)
    kwargs = vars(parser.parse_args())
    configOverride = getConfigOverrideFromParser(kwargs, ProgressiveGANTrainer._defaultConfig)

    if kwargs['overrides']:
        parser.print_help()
        sys.exit()

    # configuration file path
    config_file_path = kwargs.get("configPath", None)
    config = load_config_file(config_file_path)

    ########### MODEL CONFIG ############
    model_config = config["modelConfig"]
    architecture = baseArgs.model_name
    model_config['arch'] = architecture

    for item, val in configOverride.items():
        model_config[item] = val

    ########### DATA CONFIG #############
    data_config  = config.get("dataConfig", {})
    for item, val in configOverride.items():
        data_config[item] = val

    exp_name = config.get("name", "default")
    checkPointDir = data_config["output_path"]
    # check if model exists
    if not os.path.exists(os.path.join(data_config["output_path"], exp_name)) and \
        exp_name not in data_config["output_path"]:
        config['date'] = time.strftime("%Y-%m-%d %H:%M:%S")
        # set experiment name
        exp_name += "_" + architecture + "_" + time.strftime('%Y_%m_%d')
        config["name"] = exp_name
    
        # create output folders
        
        checkPointDir = mkdir_in_path(checkPointDir, data_config["transform"])
        checkPointDir = mkdir_in_path(checkPointDir, architecture)
        checkPointDir = mkdir_in_path(checkPointDir, exp_name)
        data_config["output_path"] = checkPointDir
        config["dataConfig"] = data_config
        print(f"Creating dir {checkPointDir}")

    # LOAD CHECKPOINT
    print("Search and load last checkpoint")
    checkPointData = getLastCheckPoint(checkPointDir, exp_name)
    nSamples = kwargs['n_samples']

    # CONFIG DATA MANAGER
    print("Data manager configuration")
    data_manager = AudioDataManager(**data_config)
    model_config['output_shape'] = data_manager.get_output_shape()
    config["modelConfig"] = model_config

    # CONFIG VISUALIZATION
    vis_module = None
    if baseArgs.np_vis:
        vis_module = importlib.import_module("models.visualization.np_visualizer")
    elif baseArgs.no_vis:
        print("Visualization disabled")
    else:
        vis_output_dir = mkdir_in_path(checkPointDir, 'plots')
        # vis_module = importlib.import_module("visualization.visualizer")
        vis_module = importlib.import_module("models.visualization.visualization_manager")
        visual_config = data_config.get('visual_config')
        
        vis_manager = \
            getVisualizer(data_config.get('transform', 'waveform'))(
                output_path=vis_output_dir,
                env=exp_name,
                sampleRate=data_config.get('sample_rate'),
                **visual_config)

        vis_manager.lossVisualizer = LossVisualizer(
            output_path=vis_output_dir,
            env=exp_name)

    # Audio rendering module
    audio_module = None
    if baseArgs.render_audio:
        audio_module = importlib.import_module("audio.tools")

    print("Running " + baseArgs.model_name)

    # Add commit to config file
    repo = git.Repo(os.path.realpath(__file__), search_parent_directories=True)
    config['commit'] = repo.head.object.hexsha

    # Save config file
    save_config_file(config,
                     os.path.join(checkPointDir, f'{exp_name}_config.json'))

    partitionValue = getVal(kwargs, "partition_value",
                            config.get("partitionValue", None))

    set_benchmark = config.get('benchmark', True)
    print(f"Setting cudnn benchmark to {set_benchmark}")
    cudnn.benchmark = set_benchmark

    GANTrainer = trainerModule(pathdb=data_config["output_path"],
                               useGPU=GPU_is_available(),
                               visualisation=vis_manager,
                               audioRender=audio_module,
                               dataManager=data_manager,
                               dataConfig=data_config,
                               lossIterEvaluation=kwargs["evalIter"],
                               checkPointDir=checkPointDir,
                               saveIter= kwargs["saveIter"],
                               modelLabel=exp_name,
                               partitionValue=partitionValue,
                               nSamples=nSamples,
                               dataType=data_config.get('data_type', None),
                               config=model_config)

    # If a checkpoint is found, load it
    if not kwargs["restart"] and checkPointData is not None:
        trainConfig, pathModel, pathTmpData = checkPointData
        GANTrainer.loadSavedTraining(pathModel, trainConfig, pathTmpData)

    GANTrainer.train()
