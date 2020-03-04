import os
import json
import pickle as pkl
import numpy as np

import torch
import ipdb

from utils.config import getConfigFromDict, getDictFromConfig, BaseConfig

from tqdm import tqdm, trange

from tools import checkexists_mkdir, mkdir_in_path

import logging
from time import time

# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter


class GANTrainer():
    r"""
    A class managing a progressive GAN training. Logs, chekpoints,
    visualization, and number iterations are managed here.
    """

    def __init__(self,
                 pathdb,
                 useGPU=True,
                 visualisation=None,
                 audioRender=None,
                 dataManager=None,
                 dataConfig=None,
                 lossIterEvaluation=5000,
                 lossPlot=5000,
                 saveIter=5000,
                 checkPointDir=None,
                 modelLabel="GAN",
                 config=None,
                 pathAttribDict=None,
                 selectedAttributes=None,
                 imagefolderDataset=False,
                 ignoreAttribs=False,
                 pathPartition=None,
                 partitionValue=None,
                 dataType='IMAGE',
                 nSamples=10,
                 debug=False,
                 **kargs):
        r"""
        Args:
            - pathdb (string): path to the directorty containing the image
            dataset.
            - useGPU (bool): set to True if you want to use the available GPUs
            for the training procedure
            - visualisation (module): if not None, a visualisation module to
            follow the evolution of the training
            - lossIterEvaluation (int): size of the interval on which the
            model's loss will be evaluated
            - saveIter (int): frequency at which at checkpoint should be saved
            (relevant only if modelLabel != None)
            - checkPointDir (string): if not None, directory where the
            checkpoints should be saved
            - modelLabel (string): name of the model
            - config (dictionary): configuration dictionnary.
            for all the possible options
            - pathAttribDict (string): path to the attribute dictionary giving
                                       the labels of the dataset
            - selectedAttributes (list): if not None, consider only the listed
                                     attributes for labelling
            - imagefolderDataset (bool): set to true if the data are stored in
                                        the fashion of a
                                        torchvision.datasests.ImageFolderDataset
                                        object
            - ignoreAttribs (bool): set to True if the input attrib dict should
                                    only be used as a filter on image's names
            - pathPartition (string): if only a subset of the original dataset
                                      should be used
            - pathValue (string): partition value
        """

        # Parameters
        # Training dataset
        self.debug = debug
        self.path_db = pathdb
        self.pathPartition = pathPartition
        self.partitionValue = partitionValue
        self.dataType = dataType

        if config is None:
            config = {}

        # Load the training configuration
        self.readTrainConfig(config)

        # Checkpoints ?
        self.checkPointDir = checkPointDir
        self.modelLabel = modelLabel
        self.saveIter = saveIter
        self.pathLossLog = None
        self.nSamples = nSamples

        if self.checkPointDir is not None:
            self.pathLossLog = os.path.abspath(os.path.join(self.checkPointDir,
                                                            self.modelLabel
                                                            + '_losses.pkl'))
            self.pathRefVector = os.path.abspath(os.path.join(self.checkPointDir,
                                                              self.modelLabel
                                                              + '_refVectors.pt'))
       # Visualization
        self.visualisation = visualisation
        self.lossVisualizer = visualisation.lossVisualizer
        self.lossVisualizer.publish_config_file(config)

        # Initialize the model
        self.useGPU = useGPU

        if not self.useGPU:
            self.numWorkers = 1

        # self.pathAttribDict = pathAttribDict
        self.pathAttribDict = config.get("pathAttribDict", None)
        self.selectedAttributes = selectedAttributes
        self.imagefolderDataset = imagefolderDataset
        self.modelConfig.attribKeysOrder = None

        self.dataManager = dataManager
        self.dataConfig = dataConfig

        self.startScale = self.modelConfig.startScale

        # CONDITIONAL GAN
        if not ignoreAttribs:
            self.modelConfig.attribKeysOrder = self.dataManager.loader.getKeyOrders()
            print("AC-GAN classes : ")
            print(self.modelConfig.attribKeysOrder)
            print("")

        # Intern state
        self.runningLoss = {}
        self.lossPlot = lossPlot
        # self.startScale = 0

        self.startIter = 0
        self.lossProfile = []
        self.epoch = 0
        # print("%d images detected" % int(len(self.getDataset(0, size=10))))
        self.initModel()
        # Audio rendering
        self.audioRender = audioRender
        self.root_output_dir = mkdir_in_path(checkPointDir, 'output')

        # self.nDataVisualization = 16
        self.nDataVisualization = nSamples
            
        self.refVectorVisualization, self.refVectorLabels = \
            self.model.buildNoiseData(self.nDataVisualization)
        # Loss printing
        self.lossIterEvaluation = lossIterEvaluation

        print(f"Setting logging file")
        self.logger = logging.getLogger()
        fhandler = logging.FileHandler(filename=f'{checkPointDir}/{self.modelLabel}.log', mode='a')
        formatter = logging.Formatter('%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s')
        fhandler.setFormatter(formatter)
        self.logger.addHandler(fhandler)
        self.logger.setLevel(logging.INFO)

        self.writer = SummaryWriter(mkdir_in_path(checkPointDir, 'runs'))


    def initModel(self):
        r"""
        Initialize the GAN model.
        """
        pass

    def updateRunningLosses(self, allLosses):
        for name, value in allLosses.items():

            if name not in self.runningLoss:
                self.runningLoss[name] = [0, 0]

            self.runningLoss[name][0] += value
            self.runningLoss[name][1] += 1

    def resetRunningLosses(self):

        self.runningLoss = {}

    def updateLossProfile(self, iter):

        nPrevIter = len(self.lossProfile[-1]["iter"])
        self.lossProfile[-1]["iter"].append(iter)

        newKeys = set(self.runningLoss.keys())
        existingKeys = set(self.lossProfile[-1].keys())

        toComplete = existingKeys - newKeys

        for item in newKeys:
            if item not in existingKeys:
                self.lossProfile[-1][item] = [None for x in range(nPrevIter)]

            value, stack = self.runningLoss[item]
            self.lossProfile[-1][item].append(value /float(stack))

        for item in toComplete:
            if item in ["scale", "iter"]:
                continue
            self.lossProfile[-1][item].append(None)

    def readTrainConfig(self, config):
        r"""
        Load a permanent configuration describing a models. The variables
        described in this file are constant through the training.
        """
        self.modelConfig = BaseConfig()
        getConfigFromDict(self.modelConfig, config, self.getDefaultConfig())


    def loadSavedTraining(self, pathModel, pathTrainConfig, pathTmpConfig, loadGOnly=False, loadDOnly=False, finetune=False):
        r"""
        Load a given checkpoint.

        Args:

            - pathModel (string): path to the file containing the model
                                 structure (.pt)
            - pathTrainConfig (string): path to the reference configuration
                                        file of the training. WARNING: this
                                        file must be compatible with the one
                                        pointed by pathModel
            - pathTmpConfig (string): path to the temporary file describing the
                                      state of the training when the checkpoint
                                      was saved. WARNING: this file must be
                                      compatible with the one pointed by
                                      pathModel
        """

        # Load the temp configuration
        tmpPathLossLog = None
        tmpConfig = {}

        if pathTmpConfig is not None:
            tmpConfig = json.load(open(pathTmpConfig, 'rb'))
            self.startScale = tmpConfig["scale"]
            self.startIter = tmpConfig["iter"]
            self.runningLoss = tmpConfig.get("runningLoss", {})

            tmpPathLossLog = tmpConfig.get("lossLog", None)

        if tmpPathLossLog is None:
            self.lossProfile = [
                {"iter": [], "scale": self.startScale}]
        elif not os.path.isfile(tmpPathLossLog):
            print("WARNING : couldn't find the loss logs at " +
                  tmpPathLossLog + " resetting the losses")
            self.lossProfile = [
                {"iter": [], "scale": self.startScale}]
        else:
            self.lossProfile = pkl.load(open(tmpPathLossLog, 'rb'))
            self.lossProfile = self.lossProfile[:(self.startScale + 1)]
            if self.lossProfile[-1]["iter"][-1] > self.startIter:
                indexStop = next(x[0] for x in enumerate(self.lossProfile[-1]["iter"])
                                 if x[1] > self.startIter)
                self.lossProfile[-1]["iter"] = self.lossProfile[-1]["iter"][:indexStop]

                for item in self.lossProfile[-1]:
                    if isinstance(self.lossProfile[-1][item], list):
                        self.lossProfile[-1][item] = \
                            self.lossProfile[-1][item][:indexStop]

        # Read the training configuration
        if not finetune:
            trainConfig = json.load(open(pathTrainConfig, 'rb'))
            self.readTrainConfig(trainConfig)

        # Re-initialize the model
        self.initModel()
        self.model.load(pathModel,
                        loadG=not loadDOnly,
                        loadD=not loadGOnly,
                        finetuning=finetune)

        # Build retrieve the reference vectors
        self.refVectorPath = tmpConfig.get("refVectors", None)
        if self.refVectorPath is None:
            self.refVectorVisualization, self.refVectorLabels = \
                self.model.buildNoiseData(self.nDataVisualization)
        elif not os.path.isfile(self.refVectorPath):
            print("WARNING : no file found at " + self.refVectorPath
                  + " building new reference vectors")
            self.refVectorVisualization, self.refVectorLabels = \
                self.model.buildNoiseData(self.nDataVisualization, skipAtts=True)
        else:
            self.refVectorVisualization = torch.load(
                open(self.refVectorPath, 'rb'),
                map_location="cuda" if self.useGPU else "cpu")

    def getDefaultConfig(self):
        pass

    def resetVisualization(self, nDataVisualization):

        self.nDataVisualization = nDataVisualization
        self.refVectorVisualization, self.refVectorLabels = \
            self.model.buildNoiseData(self.nDataVisualization)

    def saveBaseConfig(self, outPath):
        r"""
        Save the model basic configuration (the part that doesn't change with
        the training's progression) at the given path
        """

        outConfig = getDictFromConfig(
            self.modelConfig, self.getDefaultConfig())

        if "alphaJumpMode" in outConfig:
            if outConfig["alphaJumpMode"] == "linear":

                outConfig.pop("iterAlphaJump", None)
                outConfig.pop("alphaJumpVals", None)

        with open(outPath, 'w', encoding='utf-8') as fp:
            json.dump(outConfig, fp, indent=4)

    def saveCheckpoint(self,
                       outDir, 
                       outLabel, 
                       scale, 
                       iter):
        r"""
        Save a checkpoint at the given directory. Please note that the basic
        configuration won't be saved.

        This function produces 2 files:
        outDir/outLabel_tmp_config.json -> temporary config
        outDir/outLabel -> networks' weights

        And update the two followings:
        outDir/outLabel_losses.pkl -> losses util the last registered iteration
        outDir/outLabel_refVectors.pt -> reference vectors for visualization
        """
        pathModel = os.path.join(outDir, outLabel + ".pt")
        self.model.save(pathModel)

        # Tmp Configuration
        pathTmpConfig = os.path.join(outDir, outLabel + "_tmp_config.json")
        outConfig = {'scale': scale,
                     'iter': iter,
                     'lossLog': self.pathLossLog,
                     'refVectors': self.pathRefVector,
                     'runningLoss': self.runningLoss}

        # Save the reference vectors
        torch.save(self.refVectorVisualization, open(self.pathRefVector, 'wb'))

        with open(pathTmpConfig, 'w') as fp:
            json.dump(outConfig, fp, indent=4)

        if self.pathLossLog is None:
            raise AttributeError("Logging mode disabled")
        else:
            pkl.dump(self.lossProfile, open(self.pathLossLog, 'wb'))

    def eval_save(self,
                  real_input,
                  real_labels,
                  scale):
        outLabel = self.modelLabel + ("_s%d_i%d" % (scale, self.iter + 1))
        inputLatent, _ = self.model.buildNoiseData(
            len(real_input), real_labels, skipAtts=True)

        ref_g = self.model.test(inputLatent)
        ref_g_smooth = self.model.test(inputLatent, True)
        name_real = f"{outLabel}_scale_{scale}_real"
        name_gen  = f"{outLabel}_scale_{scale}_gen"
        name_avg  = f"{outLabel}_scale_{scale}_avg"

    def sendLossToVisualization(self, scale):

        self.lossVisualizer.publish(data=self.lossProfile[scale],
                            name=self.modelLabel + f'loss_scale_{scale}',
                            output_dir=self.output_dir)

    def getMiniBatchSize(self, scale):

        if type(self.modelConfig.miniBatchSize) is list:
            try:
                return self.modelConfig.miniBatchSize[scale]
            except Exception as e:
                return self.modelConfig.miniBatchSize[-1]
        else:
            return self.modelConfig.miniBatchSize


    def getDBLoader(self, scale):
        r"""
        Load the training dataset for the given scale.

        Args:

            - scale (int): scale at which we are working

        Returns:

            A dataset with properly resized inputs.
        """
        dataset = self.getDataset(scale)
        bsize = self.getMiniBatchSize(scale)

        print(f"Using batch of size {bsize}")
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=bsize,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=self.dataConfig.get("num_workers", 0))

    def getDataset(self, size=None, audio=False):

        self.visualisation.set_postprocessing(self.data_manager.post_pipeline)
        loader = self.data_manager.get_loader()

        return loader

    def inScaleUpdate(self, iter, scale, inputs_real):
        return inputs_real


    def loss_names_to_code(self, key):
        name2code = {
            'lossD_classif': 'D_cls',
            'lossG_classif': 'G_cls',
            'lossD': 'D',
            'lossD': 'G',
            'lossD_real': 'D_real',
            'lossD_fake': 'D_fake',
            'lossD_Grad': 'D_grad',
            'lossG_Grad': 'G_grad',
            'lipschitz_norm': 'lipn',
            'lossD_Epsilon': 'D_eps',
            'lossG_fake': 'G_fake',
            'lossG_GDPP': 'GDPP'

        }
        return name2code.get(key, key)


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
            for inputs_real, labels in t:
                if inputs_real.size()[0] < self.getMiniBatchSize(scale):
                    continue

                evaluate = (self.iter + 1) % (self.lossIterEvaluation) == 0 \
                            and self.visualisation is not None

                if evaluate: self.model.register_grads = True

                # Additionnal updates inside a scale
                inputs_real = self.inScaleUpdate(self.iter, scale, inputs_real)
                # Optimize parameters
                # if len(data) > 2:
                if type(labels) is tuple:
                    # mask = data[2]
                    mask = labels[1]
                    allLosses = self.model.optimizeParameters(
                        inputs_real, inputLabels=labels, inputMasks=mask)
                else:
                    allLosses = self.model.optimizeParameters(inputs_real,
                                                              inputLabels=labels,
                                                              fakeLabels=dbLoader.dataset.get_labels(inputs_real.size(0)))

                # Update and print losses
                self.updateRunningLosses(allLosses)
                state_msg = f'Iter: {self.iter}; scale: {scale} '
                for key, val in allLosses.items():
                    state_msg += f'{self.loss_names_to_code(key)}: {val:.2f}; '
                t.set_description(state_msg)

                # Plot losses
                if self.iter % self.lossPlot == 0 and self.iter != 0:
                    # Reinitialize the losses
                    self.updateLossProfile(self.iter)
                    self.resetRunningLosses()
                    self.sendLossToVisualization(scale)

                # Evaluation and visualize 
                if evaluate:
                    self.evaluation(scale)
                # Save checkpoint
                if self.checkPointDir is not None and self.iter % (self.saveIter - 1) == 0 and self.iter != 0:
                    labelSave = self.modelLabel + ("_s%d_i%d" % (scale, self.iter))
                    # Evaluate
                    self.eval_save(real_input=inputs_real,
                                   real_labels=labels,
                                   scale=scale)
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
