import os

import ipdb

from .progressive_gan_trainer import ProgressiveGANTrainer

from .transform_style_gan import TStyleGAN
from tqdm import trange
from .spgan_config import _C
import numpy as np

class TStyleGANTrainer(ProgressiveGANTrainer):
    r"""
    A class managing a progressive GAN training. Logs, chekpoints, visualization,
    and number iterations are managed here.
    """
    _defaultConfig = _C

    def getDefaultConfig(self):
        return TStyleGANTrainer._defaultConfig

    def __init__(self, **kwargs):
        r"""
        Args:
            - pathdb (string): path to the directorty containing the image
                               dataset
            - useGPU (bool): set to True if you want to use the available GPUs
                             for the training procedure
            - visualisation (module): if not None, a visualisation module to
                                      follow the evolution of the training
            - lossIterEvaluation (int): size of the interval on which the
                                        model'sloss will be evaluated
            - saveIter (int): frequency at which at checkpoint should be saved
                              (relevant only if modelLabel != None)
            - checkPointDir (string): if not None, directory where the checkpoints
                                      should be saved
            - modelLabel (string): name of the model
            - config (dictionary): configuration dictionnary. See std_p_gan_config.py
                                   for all the possible options
            - numWorkers (int): number of GOU to use. Will be set to one if not
                                useGPU
            - stopOnShitStorm (bool): should we stop the training if a diverging
                                     behavior is detected ?
        """

        ProgressiveGANTrainer.__init__(self, **kwargs)

    def initModel(self):
        r"""
        Initialize the GAN model.
        """
        print("Init StyleGAN")
        config = self.initScaleShapes()
        self.model = TStyleGAN(useGPU=self.useGPU, **config)
