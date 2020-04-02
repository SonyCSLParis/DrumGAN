import os
import json
import pickle as pkl
import numpy as np

import torch

from utils.config import get_config_from_dict, get_dict_from_config, BaseConfig
from utils.utils import mkdir_in_path
from abc import ABC, abstractmethod

from tqdm import tqdm, trange

from time import time
import ipdb
from visualization import LossVisualizer

class GANTrainer():
    r"""
    A class managing a progressive GAN training. Logs, chekpoints,
    visualization, and number iterations are managed here.
    """

    def __init__(self,
                 model_name,
                 checkpoint_dir,
                 gpu=True,
                 visualisation=None,
                 loader=None,
                 loss_plot_i=5000,
                 eval_i=5000,
                 saveIter=5000,
                 config=None,
                 pathAttribDict=None,
                 selectedAttributes=None,
                 ignoreAttribs=False,
                 n_samples=10,
                 save_gen=True,
                 vis_manager=None,
                 **kargs):
        r"""
        Args:
            - pathdb (string): path to the directorty containing the image
            dataset.
            - gpu (bool): set to True if you want to use the available GPUs
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
            - pathValue (string): partition value
        """

        # Parameters
        if config is None:
            config = {}

        # Load the training configuration
        self.readTrainConfig(config)

        # Checkpoints ?
        assert os.path.exists(checkpoint_dir), f'Checkpoint  dir {checkpoint_dir} does not exist!'
        self.checkPointDir = checkpoint_dir
        self.output_dir = mkdir_in_path(self.checkPointDir, 'output')
        self.modelLabel = model_name
        self.saveIter = save_iter
        self.pathLossLog = None
        self.nSamples = n_samples
        self.save_gen = save_gen
        if self.checkPointDir is not None:
            self.pathLossLog = os.path.abspath(os.path.join(self.checkPointDir,
                                                            self.modelLabel
                                                            + '_losses.pkl'))
            self.pathRefVector = os.path.abspath(os.path.join(self.checkPointDir,
                                                              self.modelLabel
                                                              + '_refVectors.pt'))

        # Initialize the model
        self.useGPU = gpu
        self.device = torch.device('cuda' if self.useGPU else 'cpu')

        if not self.useGPU:
            self.numWorkers = 1

        self.loader = loader

        self.startScale = self.modelConfig.startScale

        # # CONDITIONAL GAN
        if self.modelConfig.ac_gan:
            self.modelConfig.attribKeysOrder = \
                self.loader.get_attribute_dict()
        # Intern state
        self.runningLoss = {}
        
        # self.startScale = 0

        self.startIter = 0
        self.lossProfile = []
        self.epoch = 0
        # print("%d images detected" % int(len(self.getDataset(0, size=10))))
        self.initModel()
        # Loss printing
        self.loss_plot_i = loss_plot_i
        self.eval_i = eval_i
        self.loss_visualizer = \
            LossVisualizer(output_path=self.output_dir,
                           env=self.modelLabel,
                           save_figs=True)

        # init ref eval vectors
        self.init_reference_eval_vectors()
        self.vis_manager = vis_manager

    @abstractmethod
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
        get_config_from_dict(self.modelConfig, config, self.getDefaultConfig())


    def load_saved_training(self, pathModel, pathTrainConfig, pathTmpConfig, loadGOnly=False, loadDOnly=False, finetune=False):
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

        try:
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
        except:
            self.lossProfile = [
                {"iter": [], "scale": self.startScale}]

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

        outConfig = get_dict_from_config(
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

        with open(pathTmpConfig, 'w') as fp:
            json.dump(outConfig, fp, indent=4)

        if self.pathLossLog is None:
            raise AttributeError("Logging mode disabled")
        else:
            pkl.dump(self.lossProfile, open(self.pathLossLog, 'wb'))


    def getMiniBatchSize(self, scale):

        if type(self.modelConfig.miniBatchSize) is list:
            try:
                return self.modelConfig.miniBatchSize[scale]
            except Exception as e:
                return self.modelConfig.miniBatchSize[-1]
        else:
            return self.modelConfig.miniBatchSize


    def getDBLoader(self, scale=0):
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
                                           num_workers=getattr(self, "num_workers", 0))

    def getDataset(self):
        raise NotImplementedError

    def inScaleUpdate(self, iter, scale, inputs_real):
        return inputs_real

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
                # if inputs_real.size()[0] < self.getMiniBatchSize(scale):
                #     continue

                # Additionnal updates inside a scale
                inputs_real = self.inScaleUpdate(self.iter, scale, inputs_real)
                # Optimize parameters
                allLosses = self.model.optimizeParameters(
                    inputs_real, 
                    inputLabels=labels,
                    fakeLabels=self.loader.get_random_labels(inputs_real.size(0)))
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

    def publish_loss(self):
        self.loss_visualizer.publish(self.lossProfile[-1])

    def init_reference_eval_vectors(self, batch_size=50):

        self.true_ref, self.ref_labels = self.loader.get_validation_set(batch_size)
        self.ref_labels_str = self.loader.index_to_labels(self.ref_labels, transpose=True)

        batch_size = min(batch_size, len(self.ref_labels))
        if self.modelConfig.ac_gan:
            self.ref_z, _ = self.model.buildNoiseData(batch_size, inputLabels=self.ref_labels)
        else:
            self.ref_z, _ = self.model.buildNoiseData(batch_size)

    def test_GAN(self):
        # sample fake data
        fake = self.model.test_G(input=self.ref_z, getAvG=False, toCPU=not self.useGPU)
        fake_avg = self.model.test_G(input=self.ref_z, getAvG=True, toCPU=not self.useGPU)
        
        # predict labels for fake data
        D_fake, fake_emb = self.model.test_D(fake, output_device='cpu')
        D_fake = self.loader.index_to_labels(D_fake.detach(), transpose=True)

        D_fake_avg, fake_avg_emb = self.model.test_D(fake_avg, output_device='cpu')
        D_fake_avg = self.loader.index_to_labels(D_fake_avg.detach(), transpose=True)
        
        # predict labels for true data
        true, _ = self.loader.get_validation_set(len(self.ref_labels), process=True)
        D_true, true_emb = self.model.test_D(true, output_device='cpu')
        D_true = self.loader.index_to_labels(D_true.detach(), transpose=True)

        return D_true, true_emb.detach(), \
               D_fake, fake_emb.detach(), \
               D_fake_avg, fake_avg_emb.detach(), \
               true, fake.detach(), fake_avg.detach()

    def run_tests_evaluation_and_visualization(self, scale):
        scale_output_dir = mkdir_in_path(self.output_dir, f'scale_{scale}')
        iter_output_dir  = mkdir_in_path(scale_output_dir, f'iter_{self.iter}')
        from utils.utils import saveAudioBatch

        D_true, true_emb, \
        D_fake, fake_emb, \
        D_fake_avg, fake_avg_emb, \
        true, fake, fake_avg = self.test_GAN()
        
        if self.modelConfig.ac_gan:
            output_dir = mkdir_in_path(iter_output_dir, 'classification_report')
            if not hasattr(self, 'cls_vis'):
                from visualization.visualization import AttClassifVisualizer
                self.cls_vis = AttClassifVisualizer(
                    output_path=output_dir,
                    env=self.modelLabel,
                    save_figs=True,
                    attributes=self.loader.header['attributes'].keys(),
                    att_val_dict=self.loader.header['attributes'])
            self.cls_vis.output_path = output_dir
            self.cls_vis.publish(
                self.ref_labels_str, 
                D_true,
                name=f'{scale}_true',
                title=f'scale {scale} True data')
            
            self.cls_vis.publish(
                self.ref_labels_str, 
                D_true,
                name=f'{scale}_fake',
                title=f'scale {scale} Fake data')

        if self.save_gen:
            output_dir = mkdir_in_path(iter_output_dir, 'generation')
            saveAudioBatch(
                self.loader.postprocess(fake), 
                path=output_dir, 
                basename=f'gen_audio_scale_{scale}')

        if self.vis_manager != None:
            output_dir = mkdir_in_path(iter_output_dir, 'audio_plots')
            self.vis_manager.set_postprocessing(
                self.loader.get_postprocessor())
            self.vis_manager.publish(
                true[:5], 
                labels=D_true[0][:5], 
                name=f'real_scale_{scale}', 
                output_dir=output_dir)
            self.vis_manager.publish(
                fake[:5], 
                labels=D_fake[0][:5], 
                name=f'fake_scale_{scale}', 
                output_dir=output_dir)
