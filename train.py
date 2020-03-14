print("Running training script...")
import os
import sys
import importlib
import argparse

import time

from utils.utils import getVal, getLastCheckPoint, loadmodule, GPU_is_available, checkexists_mkdir, mkdir_in_path
from utils.config import getConfigOverrideFromParser, updateParserWithConfig

from pg_gan.progressive_gan_trainer import ProgressiveGANTrainer
from data.preprocessing import AudioPreprocessor
from data.nsynth import NSynthLoader
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
    print("")

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
                        type=int, dest="saveIter", default=25000)
    parser.add_argument('-l', '--loss_iter', help="If it applies, frequence at\
                        which a checkpoint should be saved. In the case of a\
                        evaluation test, iteration to work on.",
                        type=int, dest="lossIter", default=5000)
    parser.add_argument('--seed', action='store_true', help="Partition's value")
    parser.add_argument('--n_samples', type=int, default=10, help="Partition's value")
    
    # Parse command line args
    baseArgs, unknown = parser.parse_known_args()

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
    for item, val in configOverride.items():
        model_config[item] = val

    ########### DATA CONFIG #############
    for item, val in configOverride.items():
        data_config[item] = val

    exp_name = config.get("name", "default")
    checkPointDir = config["output_path"]
    checkPointDir = mkdir_in_path(checkPointDir, exp_name)
    # config["output_path"] = checkPointDir

    # LOAD CHECKPOINT
    print("Search and load last checkpoint")
    checkPointData = getLastCheckPoint(checkPointDir, exp_name)
    nSamples = kwargs['n_samples']

    # CONFIG DATA MANAGER
    print("Data manager configuration")
    data_manager = AudioPreprocessor(**config['transformConfig'])

    data_loader = NSynthLoader(dbname=f"NSynth_{data_manager.transform}",
                               output_path=checkPointDir,
                               preprocessing=data_manager.get_preprocessor(),
                               **config['loaderConfig'])

    print(f"Loading data. Found {len(data_loader)} instances")
    model_config['output_shape'] = data_manager.get_output_shape()
    config["modelConfig"] = model_config

    # Save config file
    save_config_file(config, os.path.join(checkPointDir, f'{exp_name}_config.json'))

    GANTrainer = ProgressiveGANTrainer(
                               modelLabel=exp_name,
                               pathdb=config["output_path"],
                               useGPU=GPU_is_available(),
                               dataLoader=data_loader,
                               lossIter=baseArgs.lossIter,
                               checkPointDir=checkPointDir,
                               saveIter= baseArgs.saveIter,
                               nSamples=nSamples,
                               config=model_config)

    # If a checkpoint is found, load it
    if not kwargs["restart"] and checkPointData is not None:
        trainConfig, pathModel, pathTmpData = checkPointData
        GANTrainer.loadSavedTraining(pathModel, trainConfig, pathTmpData)

    GANTrainer.train()
