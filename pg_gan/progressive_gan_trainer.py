import os
import torch
import torch.nn
import torch.nn.functional as F

import ipdb

from .pgan_config import _C
from .progressive_gan import ProgressiveGAN
from .gan_trainer import GANTrainer
from utils.utils import getMinOccurence
from tqdm import trange

from time import time
import numpy as np
from tools import mkdir_in_path

import traceback


class ResizeWrapper():
    def __init__(self, new_size):
        self.size = new_size
    def __call__(self, image):
        assert np.argmax(self.size) == np.argmax(image.shape[-2:]), \
            f"Resize dimensions mismatch, Target shape {self.size} \
                != image shape {image.shape}"
        if type(image) is not np.ndarray:
            image = image.numpy()
        out = interpolate(torch.from_numpy(image).unsqueeze(0), size=self.size).squeeze(0)
        return out


class ProgressiveGANTrainer(GANTrainer):
    r"""
    A class managing a progressive GAN training. Logs, chekpoints, visualization,
    and number iterations are managed here.
    """
    _defaultConfig = _C

    def getDefaultConfig(self):
        return ProgressiveGANTrainer._defaultConfig

    def __init__(self,
                 pathdb,
                 miniBatchScheduler=None,
                 datasetProfile=None,
                 configScheduler=None,
                 **kwargs):
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

        self.configScheduler = {}
        if configScheduler is not None:
            self.configScheduler = {
                int(key): value for key, value in configScheduler.items()}

        self.miniBatchScheduler = {}
        if miniBatchScheduler is not None:
            self.miniBatchScheduler = {
                int(x): value for x, value in miniBatchScheduler.items()}

        self.datasetProfile = {}
        if datasetProfile is not None:
            self.datasetProfile = {
                int(x): value for x, value in datasetProfile.items()}

        self.postprocessors = []

        GANTrainer.__init__(self, pathdb, **kwargs)

        resize_inv = ResizeWrapper(self.outputShapes[-1])
        self.visualisation.set_postprocessing(self.dataManager.get_post_processor(insert_transform=resize_inv))


    def initScaleShapes(self):
        
        config = {key: value for key, value in vars(self.modelConfig).items()}
        config["depthScale0"] = self.modelConfig.depthScales[0]

        h_size = self.modelConfig.output_shape[-1]
        w_size = self.modelConfig.output_shape[-2]

        config['scaleShapes'] = [(int(np.ceil(w_size / df[0])), int(np.ceil(h_size / df[1]))) for df in self.modelConfig.downSamplingFactor]
        self.outputShapes = config['scaleShapes']
        config["sizeScale0"] = config['scaleShapes'][0]
        return config

    def initModel(self):
        r"""
        Initialize the GAN model.
        """
        print("Init P-GAN")
        config = self.initScaleShapes()
        self.model = ProgressiveGAN(useGPU=self.useGPU, **config)

    def readTrainConfig(self, config):
        r"""
        Load a permanent configuration describing a models. The variables
        described in this file are constant through the training.
        """

        GANTrainer.readTrainConfig(self, config)
        if self.modelConfig.alphaJumpMode not in ["custom", "linear"]:
            raise ValueError(
                "alphaJumpMode should be one of the followings: \
                'custom', 'linear'")

        if self.modelConfig.alphaJumpMode == "linear":

            self.modelConfig.alphaNJumps[0] = 0
            self.modelConfig.iterAlphaJump = []
            self.modelConfig.alphaJumpVals = []

            self.updateAlphaJumps(
                self.modelConfig.alphaNJumps, self.modelConfig.alphaSizeJumps)

        self.scaleSanityCheck()

    def scaleSanityCheck(self):

        # Sanity check
        n_scales = min(len(self.modelConfig.depthScales),
                       len(self.modelConfig.maxIterAtScale),
                       len(self.modelConfig.iterAlphaJump),
                       len(self.modelConfig.alphaJumpVals))

        self.modelConfig.depthScales = self.modelConfig.depthScales[:n_scales]
        self.modelConfig.maxIterAtScale = self.modelConfig.maxIterAtScale[:n_scales]
        self.modelConfig.iterAlphaJump = self.modelConfig.iterAlphaJump[:n_scales]
        self.modelConfig.alphaJumpVals = self.modelConfig.alphaJumpVals[:n_scales]

        self.modelConfig.size_scales = [4]
        for scale in range(1, n_scales):
            self.modelConfig.size_scales.append(
                self.modelConfig.size_scales[-1] * 2)

        self.modelConfig.n_scales = n_scales

    def updateAlphaJumps(self, nJumpScale, sizeJumpScale):
        r"""
        Given the number of iterations between two updates of alpha at each
        scale and the number of updates per scale, build the effective values of
        self.maxIterAtScale and self.alphaJumpVals.

        Args:

            - nJumpScale (list of int): for each scale, the number of times
                                        alpha should be updated
            - sizeJumpScale (list of int): for each scale, the number of
                                           iterations between two updates
        """

        n_scales = min(len(nJumpScale), len(sizeJumpScale))

        for scale in range(n_scales):

            self.modelConfig.iterAlphaJump.append([])
            self.modelConfig.alphaJumpVals.append([])

            if nJumpScale[scale] == 0:
                self.modelConfig.iterAlphaJump[-1].append(0)
                self.modelConfig.alphaJumpVals[-1].append(0.0)
                continue

            diffJump = 1.0 / float(nJumpScale[scale])
            currVal = 1.0
            currIter = 0

            while currVal > 0:

                self.modelConfig.iterAlphaJump[-1].append(currIter)
                self.modelConfig.alphaJumpVals[-1].append(currVal)

                currIter += sizeJumpScale[scale]
                currVal -= diffJump

            self.modelConfig.iterAlphaJump[-1].append(currIter)
            self.modelConfig.alphaJumpVals[-1].append(0.0)

    def inScaleUpdate(self, iter, scale, input_real):
        if self.indexJumpAlpha < len(self.modelConfig.iterAlphaJump[scale]):

            if iter == self.modelConfig.iterAlphaJump[scale][self.indexJumpAlpha]:
                alpha = self.modelConfig.alphaJumpVals[scale][self.indexJumpAlpha]
                self.model.updateAlpha(alpha)
                self.indexJumpAlpha += 1

        if self.model.config.alpha > 0:
            low_res_real = F.adaptive_avg_pool2d(input_real, output_size=self.outputShapes[scale])
            low_res_real = F.interpolate(low_res_real, size=input_real.size()[-2:], mode='nearest')

            alpha = self.model.config.alpha
            input_real = alpha * low_res_real + (1-alpha) * input_real
        return input_real

    def updateDatasetForScale(self, scale):
        self.modelConfig.miniBatchSize = \
            getMinOccurence(self.miniBatchScheduler, scale, self.modelConfig.miniBatchSize)
        self.path_db = getMinOccurence(self.datasetProfile, scale, self.path_db)

        # Scale scheduler
        if self.configScheduler is not None and scale in self.configScheduler:
            print("Scale %d, updating the training configuration" % scale)
            self.model.updateConfig(self.configScheduler[scale])

    def addStartingScales(self):
        r"""
        If a starting scale is defined other than 0, this method adds to the network
        the first 0-startingScale blocks.
        """
        # We check len(self.model.config.depthOtherScales) == 0 so that we don't add
        # new layers to a model loaded from a checkpoint
        if self.startScale > 0 and len(self.model.config.depthOtherScales) == 0:
            for scale in range(self.startScale):
                self.model.addScale(self.modelConfig.depthScales[scale + 1])

    def train(self):
        r"""
        Launch the training. This one will stop if a divergent behavior is
        detected.

        Returns:

            - True if the training completed
            - False if the training was interrupted due to a divergent behavior
        """
        self.n_scales = len(self.modelConfig.depthScales)
        if self.checkPointDir is not None:
            pathBaseConfig = os.path.join(self.checkPointDir, self.modelLabel
                                          + "_train_config.json")
            self.saveBaseConfig(pathBaseConfig)
        self.addStartingScales()
        
        try:
            for scale in range(self.startScale, self.n_scales):
                self.output_dir = mkdir_in_path(self.root_output_dir, f"scale_{scale}")
            
                self.logger.info(f'Training scale {scale}...')
                self.updateDatasetForScale(scale)

                while scale >= len(self.lossProfile):
                    self.lossProfile.append(
                        {"scale": scale, "iter": []})

                dbLoader = self.getDBLoader(scale)
                n_batches_db = len(dbLoader)

                self.logger.info(f'Database number of batches: {n_batches_db}')
                self.logger.info(f'Number of parameters: {self.model.countParams()}')

                self.iter = 0
                if self.startIter > 0:
                    self.iter = self.startIter
                    self.startIter = 0

                shiftAlpha = 0
                while shiftAlpha < len(self.modelConfig.iterAlphaJump[scale]) and \
                        self.modelConfig.iterAlphaJump[scale][shiftAlpha] < self.iter:
                    shiftAlpha += 1

                n_iter = int(np.ceil(self.modelConfig.maxIterAtScale[scale] / n_batches_db))
                pbar = trange(n_iter,
                              desc='scale-loop')
                for scale_iter in pbar:
                    if self.iter >= self.modelConfig.maxIterAtScale[scale]: break

                    self.indexJumpAlpha = shiftAlpha

                    t1 = time()
                    status = self.trainOnEpoch(dbLoader,
                                               scale,
                                               maxIter=self.modelConfig.maxIterAtScale[scale])
                    
                    if not status:
                        return False
                    # self.iter += n_batches_db
                    while shiftAlpha < len(self.modelConfig.iterAlphaJump[scale]) and \
                            self.modelConfig.iterAlphaJump[scale][shiftAlpha] < self.iter:
                        shiftAlpha += 1
                    
                    # self.updateLossProfile(scale_iter)
                    try:        
                        state_msg = (f'Epoch: {self.epoch}; G: {self.runningLoss["lossG"][0]:.3f}; D: {self.runningLoss["lossD"][0]:.3f};'
                            f' Scale: {scale}')
                        pbar.set_description(state_msg)
                    except KeyError as k:
                        print(k)
                        pass
                    
                    self.resetRunningLosses()

                if scale == self.n_scales - 1:
                    break
                else:
                    self.model.addScale(self.modelConfig.depthScales[scale + 1])

        except KeyboardInterrupt as e:
            self.logger.info(f'Aborting training')
            print("ABORTING TRAINING")
            print("")
            print(e)
            print(traceback.print_tb(e.__traceback__))
            print("")

            if self.iter > 0:
                print("")
                print("Saving checkpoint...")
                print("")
                self.updateLossProfile(self.iter)
                checkp_name = self.modelLabel + ("_s%d_i%d" % (scale, self.iter))
                self.saveCheckpoint(outDir=self.checkPointDir,
                                    outLabel=checkp_name,
                                    scale=scale,
                                    iter=self.iter)
        except Exception as e:

            if self.debug: raise e

            self.logger.info(f'Aborting training')
            print("ABORTING TRAINING")
            print("")
            print(e)
            print(traceback.print_tb(e.__traceback__))
            print("")

            if self.iter > 0:
                print("")
                print("Saving checkpoint...")
                print("")
                self.updateLossProfile(self.iter)
                checkp_name = self.modelLabel + ("_s%d_i%d" % (scale, self.iter))
                self.saveCheckpoint(outDir=self.checkPointDir,
                                    outLabel=checkp_name,
                                    scale=scale,
                                    iter=self.iter)
        self.logger.info(f'\n{self.model.getOriginalG()}\n')
        self.logger.info(f'\n{self.model.getOriginalD()}\n')
        self.startScale = self.n_scales
        self.startIter = self.modelConfig.maxIterAtScale[-1]
        return True

    def addNewScales(self, configNewScales):

        if configNewScales["alphaJumpMode"] not in ["custom", "linear"]:
            raise ValueError("alphaJumpMode should be one of the followings: \
                            'custom', 'linear'")

        if configNewScales["alphaJumpMode"] == 'custom':
            self.modelConfig.iterAlphaJump = self.modelConfig.iterAlphaJump + \
                configNewScales["iterAlphaJump"]
            self.modelConfig.alphaJumpVals = self.modelConfig.alphaJumpVals + \
                configNewScales["alphaJumpVals"]

        else:
            self.updateAlphaJumps(configNewScales["alphaNJumps"],
                                  configNewScales["alphaSizeJumps"])

        self.modelConfig.depthScales = self.modelConfig.depthScales + \
            configNewScales["depthScales"]
        self.modelConfig.maxIterAtScale = self.modelConfig.maxIterAtScale + \
            configNewScales["maxIterAtScale"]

        self.scaleSanityCheck()

    def getDataset(self, scale, size=None):
        resize = ResizeWrapper(self.outputShapes[scale])
        self.loader = self.dataManager.get_loader()
        self.loader.set_transform(resize)
        return self.loader

