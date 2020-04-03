import os

import ipdb

from .progressive_gan_trainer import ProgressiveGANTrainer

from .transform_style_gan import TStyleGAN
from tqdm import tqdm
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

    def init_reference_eval_vectors(self, batch_size=50):

        self.true_ref, self.true_pair = self.loader.get_validation_set(batch_size)

        batch_size = min(batch_size, len(self.true_pair))
        self.ref_z, _ = self.model.buildNoiseData(batch_size)

    def trainOnEpoch(self,
                     dbLoader,
                     scale,
                     maxIter=-1):
        r"""
        Train the model on one epoch.

        Args:

            - dbLoader (DataLoader): dataset on which the training will be made
            - scale (int): scale at which is the training is performed
            - shiftIter (int): shift to apply to the iteration index when
                               looking for the next update of the alpha
                               coefficient
            - maxIter (int): if > 0, iteration at which the training should stop

        Returns:

            True if the training went smoothly
            False if a diverging behavior was detected and the training had to
            be stopped
        """
        with tqdm(dbLoader, desc='Iter-loop') as t:

            for x, y in t:
                # Additionnal updates inside a scale
                x = self.inScaleUpdate(self.iter, scale, x)
                # Optimize parameters
                allLosses = self.model.optimizeParameters(x, y, iter=self.iter)
                # Update and print losses
                self.updateRunningLosses(allLosses)
                state_msg = f'Iter: {self.iter}; scale: {scale} '
                for key, val in allLosses.items():
                    state_msg += f'{key}: {val:.2f}; '
                t.set_description(state_msg)

                # Plot losses
                if self.iter % self.loss_plot_i == 0:
                    # Reinitialize the losses
                    self.updateLossProfile(self.iter)
                    self.resetRunningLosses()
                    self.publish_loss()

                # run evaluation/tests
                if self.iter % self.eval_i == 0 and self.iter != 0:
                    self.run_tests_evaluation_and_visualization(scale)

                # Save checkpoint
                if self.iter % (self.saveIter - 1) == 0 and self.iter != 0:
                    labelSave = self.modelLabel + ("_s%d_i%d" % (scale, self.iter))
                    # Save Checkpoint
                    self.saveCheckpoint(outDir=self.checkPointDir,
                                        outLabel=labelSave,
                                        scale=scale,
                                        iter=self.iter)
                self.iter += 1
                if self.iter == maxIter:
                    return True
        self.epoch += 1
        return True