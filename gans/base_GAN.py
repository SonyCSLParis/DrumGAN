from copy import deepcopy

import torch
import torch.nn as nn


from utils.config import BaseConfig, update_config
from . import base_loss_criterions
from .ac_criterion import ACGANCriterion

from .gradient_losses import WGANGPGradientPenalty

from utils.utils import loadPartOfStateDict, finiteCheck, \
    loadStateDictCompatible, GPU_is_available

import ipdb


class BaseGAN():
    r"""Abstract class: the basic framework for GAN training.
    """
    def __init__(self,
                 dimLatentVector,
                 dimOutput=3,
                 useGPU=True,
                 baseLearningRate=0.001,
                 lossMode='WGANGP',
                 ac_gan=False,
                 attribKeysOrder=None,
                 skipAttDfake=None,
                 weightConditionD=0.0,
                 weightConditionG=0.0,
                 logisticGradReal=0.0,
                 lambdaGP=0.,
                 epsilonD=0.,
                 GDPP=False,
                 GDPP_weigth=1,
                 mix_true_fake=False,
                 mix_true_fake_scale=-1,
                 true_fake_split=-1,
                 soft_labels=False,
                 iter_d_g_ratio=-1,
                 formatLayerType="RandomZ",
                 generationActivation=None,
                 **kwargs):
        r"""
        Args:
            dimLatentVector (int): dimension of the latent vector in the model
            dimOutput (int): number of channels of the output image
            useGPU (bool): set to true if the computation should be distribued
                           in the availanle GPUs
            baseLearningRate (float): target learning rate.
            lossMode (string): loss used by the model. Must be one of the
                               following options
                              * 'MSE' : mean square loss.
                              * 'DCGAN': cross entropy loss
                              * 'WGANGP': https://arxiv.org/pdf/1704.00028.pdf
                              * 'Logistic': https://arxiv.org/pdf/1801.04406.pdf
            attribKeysOrder (dict): if not None, activate AC-GAN. In this case,
                                    both the generator and the discrimator are
                                    trained on abelled data.
            weightConditionD (float): in AC-GAN, weight of the classification
                                      loss applied to the discriminator
            weightConditionG (float): in AC-GAN, weight of the classification
                                      loss applied to the generator
            logisticGradReal (float): gradient penalty for the logistic loss
            lambdaGP (float): if > 0, weight of the gradient penalty (WGANGP)
            epsilonD (float): if > 0, penalty on |D(X)|**2
            GDPP (bool): if true activate GDPP loss https://arxiv.org/abs/1812.00068

        """
        
        # This params should go in a trainer class or similar
        self.mix_true_fake = mix_true_fake
        self.mix_true_fake_scale = mix_true_fake_scale
        self.true_fake_split = true_fake_split
        self.soft_labels = soft_labels
        self.iter_D_G_ratio = iter_d_g_ratio
        ####################################################
        if lossMode not in ['MSE', 'WGANGP', 'DCGAN', 'Logistic']:
            raise ValueError(
                "lossMode should be one of the following : ['MSE', 'WGANGP', \
                'DCGAN', 'Logistic']")

        if 'config' not in vars(self):
            self.config = BaseConfig()

        if 'trainTmp' not in vars(self):
            self.trainTmp = BaseConfig()

        self.useGPU = useGPU and torch.cuda.is_available()
        if self.useGPU:
            self.device = torch.device("cuda:0")
            self.n_devices = torch.cuda.device_count()
        else:
            self.device = torch.device("cpu")
            self.n_devices = 1
        # Latent vector dimension
        self.config.noiseVectorDim = dimLatentVector

        # Output image dimension
        self.config.dimOutput = dimOutput

        # Actual learning rate
        self.config.learningRate = baseLearningRate

        # Input formatLayer type
        self.config.formatLayerType = formatLayerType

        # AC-GAN ?
        self.config.skipAttDfake = skipAttDfake
        self.config.attribKeysOrder = deepcopy(attribKeysOrder)
        self.config.categoryVectorDim = 0
        self.config.categoryVectorDim_G = 0
        self.config.weightConditionG = weightConditionG
        self.config.weightConditionD = weightConditionD
        self.ClassificationCriterion = None
        self.config.ac_gan = ac_gan
        self.initializeClassificationCriterion()

        # GDPP
        self.config.GDPP = GDPP
        self.config.GDPP_weigth = GDPP_weigth

        self.config.latentVectorDim = self.config.noiseVectorDim \
            + self.config.categoryVectorDim_G

        # Loss criterion
        self.config.lossCriterion = lossMode
        self.lossCriterion = getattr(
            base_loss_criterions, lossMode)(self.device)

        # Overwrite generationActivation to loss mode if specified in config
        if generationActivation == "tanh":
            self.lossCriterion.generationActivation = nn.Tanh()

        # WGAN-GP
        self.config.lambdaGP = lambdaGP

        # Weight on D's output
        self.config.epsilonD = epsilonD

        # Initialize the generator and the discriminator
        self.netD = self.getNetD()
        self.netG = self.getNetG()

        # Move the networks to the gpu
        self.updateSolversDevice()

        # Logistic loss
        self.config.logisticGradReal = logisticGradReal

        # Register grads?
        self.register_grads = False


    def test_G(self, input, getAvG=False, toCPU=True, **kargs):
        r"""
        Generate some data given the input latent vector.

        Args:
            input (torch.tensor): input latent vector
        """
        input = input.to(self.device)
        if getAvG:
            if toCPU:
                return self.avgG(input).cpu()
            else:
                return self.avgG(input)
        elif toCPU:
            return self.netG(input).detach().cpu()
        else:
            return self.netG(input).detach()

    def test_D(self, input, get_labels=True, get_embeddings=True, output_device=torch.device('cuda')):
        input = input.to(self.device)
        pred, embedding = self.netD(input, True)

        if get_labels:
            pred, _ = self.ClassificationCriterion.getPredictionLabels(pred)
        if get_embeddings:
            return pred.detach().to(output_device), embedding.detach().to(output_device)
        else:
            return pred.detach().to(output_device)

    def buildAvG(self):
        r"""
        Create and upload a moving average generator.
        """
        self.avgG = deepcopy(self.getOriginalG())
        for param in self.avgG.parameters():
            param.requires_grad = False

        if self.useGPU:
            self.avgG = nn.DataParallel(self.avgG)
            self.avgG.to(device=self.device)

    def optimizeParameters(self, input_batch, inputLabels=None, **args):
        r"""
        Update the discrimator D using the given "real" inputs.

        Args:
            input (torch.tensor): input batch of real data
            inputLabels (torch.tensor): labels of the real data

        """
        allLosses = {}
        try:
            allLosses['alpha'] = self.getOriginalD().alpha
        except AttributeError:
            pass
        # Retrieve the input data
        self.real_input, self.realLabels = input_batch.to(self.device), None
        if self.config.attribKeysOrder is not None:
            self.realLabels = inputLabels.to(self.device)

        n_samples = self.real_input.size()[0]

        # Update the discriminator
        self.optimizerD.zero_grad()

        # #1 Real data
        predRealD = self.netD(self.real_input, False)
        # Classification criterion
        allLosses["lossD_classif"] = \
            self.classificationPenalty(predRealD,
                                       self.realLabels,
                                       self.config.weightConditionD,
                                       backward=True)

        lossD = self.lossCriterion.getCriterion(predRealD, True)
        allLosses["lossD_real"] = lossD.item()

        # #2 Fake data
        inputLatent, targetRandCat = self.buildNoiseData(n_samples, self.realLabels)

        predFakeG = self.netG(inputLatent).detach()
        predFakeD = self.netD(predFakeG, False)
        lossDFake = self.lossCriterion.getCriterion(predFakeD, False)
        allLosses["lossD_fake"] = lossDFake.item()
        lossD += lossDFake

        # #3 WGANGP gradient loss
        if self.config.lambdaGP > 0:
            allLosses["lossD_Grad"], allLosses["lipschitz_norm"] =\
                WGANGPGradientPenalty(input=self.real_input,
                                        fake=predFakeG,
                                        discriminator=self.netD,
                                        weight=self.config.lambdaGP,
                                        backward=True)

        # #4 Epsilon loss
        if self.config.epsilonD > 0:
            lossEpsilon = (predRealD[:, 0] ** 2).sum() * self.config.epsilonD
            lossD += lossEpsilon
            allLosses["lossD_Epsilon"] = lossEpsilon.item()


        lossD.backward(retain_graph=True)
        # finiteCheck(self.netD.module.parameters())
        finiteCheck(self.netD.parameters())
        self.optimizerD.step()

        # Logs
        lossD = 0
        for key, val in allLosses.items():

            if key.find("lossD") == 0:
                lossD += val

        allLosses["lossD"] = lossD

        # Update the generator
        self.optimizerG.zero_grad()
        self.optimizerD.zero_grad()

        # #1 Image generation
        inputNoise, targetCatNoise = self.buildNoiseData(n_samples, self.realLabels)
        predFakeG = self.netG(inputNoise)

        # #2 Status evaluation
        predFakeD, phiGFake = self.netD(predFakeG, True)

        # #2 Classification criterion
        allLosses["lossG_classif"] = \
            self.classificationPenalty(predFakeD,
                                       targetCatNoise,
                                       self.config.weightConditionG,
                                       backward=True)

        # #3 GAN criterion
        lossGFake = self.lossCriterion.getCriterion(predFakeD, True)
        allLosses["lossG_fake"] = lossGFake.item()
        lossGFake.backward()

        finiteCheck(self.getOriginalG().parameters())
        self.optimizerG.step()

        lossG = 0
        for key, val in allLosses.items():

            if key.find("lossG") == 0:
                lossG += val

        allLosses["lossG"] = lossG

        # Update the moving average if relevant
        if isinstance(self.avgG, nn.DataParallel):
            avgGparams = self.avgG.module.parameters()
        else:
            avgGparams = self.avgG.parameters()       
        
        for p, avg_p in zip(self.getOriginalG().parameters(),
                            avgGparams):
            avg_p.mul_(0.999).add_(0.001, p.data)

        return allLosses
    
    def initializeClassificationCriterion(self):
        r"""
        For labelled datasets: initialize the classification criterion.
        """
        # if self.config.weightConditionD != 0 and \
        #         not self.config.attribKeysOrder:
        #     raise AttributeError("If the weight on the conditional term isn't "
        #                          "null, then a attribute dictionnery should be"
        #                          " defined")

        # if self.config.weightConditionG != 0 and \
        #         not self.config.attribKeysOrder:
        #     raise AttributeError("If the weight on the conditional term isn't \
        #                          null, then a attribute dictionnary should be \
        #                          defined")

        if self.config.attribKeysOrder is not None and self.config.ac_gan:

            self.ClassificationCriterion = \
                    ACGANCriterion(self.config.attribKeysOrder, 
                                   skipAttDfake=self.config.skipAttDfake,
                                   soft_labels=self.soft_labels)

            # we only update the category vec dim if we are computing loss
            # on the classification (ac_gan=true). If not, we just want to instanciate
            # the ClassifCrit for accessing the label_to_indx
            if self.config.ac_gan == True:
                self.config.categoryVectorDim = \
                    self.ClassificationCriterion.getInputDim()
                
                # For G there are atts that are skipped
                self.config.categoryVectorDim_G = \
                    self.ClassificationCriterion.getInputDim(True)

    def updateSolversDevice(self, buildAvG=True):
        r"""
        Move the current networks and solvers to the GPU.
        This function must be called each time netG or netD is modified
        """
        if self.buildAvG():
        # if buildAvG:
            self.buildAvG()

        if not isinstance(self.netD, nn.DataParallel) and self.useGPU:
            self.netD = nn.DataParallel(self.netD)
        if not isinstance(self.netG, nn.DataParallel) and self.useGPU:
            self.netG = nn.DataParallel(self.netG)

        self.netD.to(self.device)
        self.netG.to(self.device)

        self.optimizerD = self.getOptimizerD()
        self.optimizerG = self.getOptimizerG()

        self.optimizerD.zero_grad()
        self.optimizerG.zero_grad()

    def buildNoiseData(self, n_samples, inputLabels=None, skipAtts=False):
        r"""
        Build a batch of latent vectors for the generator.

        Args:
            n_samples (int): number of vector in the batch
        """
        # sample random vector of length n_samples
        inputLatent = torch.randn(n_samples, self.config.noiseVectorDim).to(self.device)
        # HACK:
        # if skipAtts and all(k in self.ClassificationCriterion.skipAttDfake \
        #         for k in self.ClassificationCriterion.keyOrder):
        #     return inputLatent, None
        #################
        
        if self.config.attribKeysOrder and self.config.ac_gan:
            if inputLabels is not None:
                latentRandCat = self.ClassificationCriterion.buildLatentCriterion(inputLabels, skipAtts=skipAtts)
                targetRandCat = inputLabels
            else:
                targetRandCat, latentRandCat = \
                    self.ClassificationCriterion.buildRandomCriterionTensor(n_samples, skipAtts)

            targetRandCat = targetRandCat.to(self.device)
            latentRandCat = latentRandCat.to(self.device)
            inputLatent = torch.cat((inputLatent, latentRandCat), dim=1)

            return inputLatent, targetRandCat
        return inputLatent, None

    def buildNoiseDataWithConstraints(self, n, labels):
        constrainPart = \
            self.ClassificationCriterion.generateConstraintsFromVector(n,
                                                                       labels)
        inputLatent = torch.randn((n, self.config.noiseVectorDim, 1, 1))

        return torch.cat((inputLatent, constrainPart), dim=1)

    def getOriginalG(self):
        r"""
        Retrieve the original G network. Use this function
        when you want to modify G after the initialization
        """
        if isinstance(self.netG, nn.DataParallel):
            return self.netG.module
        return self.netG

    def getOriginalD(self):
        r"""
        Retrieve the original D network. Use this function
        when you want to modify D after the initialization
        """
        if isinstance(self.netD, nn.DataParallel):
            return self.netD.module
        return self.netD

    def getNetG(self):
        r"""
        The generator should be defined here.
        """
        pass

    def getNetD(self):
        r"""
        The discrimator should be defined here.
        """
        pass

    def getOptimizerD(self):
        r"""
        Optimizer of the discriminator.
        """
        pass

    def getOptimizerG(self):
        r"""
        Optimizer of the generator.
        """
        pass

    def getStateDict(self, saveTrainTmp=False):
        r"""
        Get the model's parameters
        """
        # Get the generator's state
        stateG = self.getOriginalG().state_dict()

        # Get the discrimator's state
        stateD = self.getOriginalD().state_dict()
        
        out_state = {'config': self.config,
                     'netG': stateG,
                     'netD': stateD}

        # Average GAN
        if isinstance(self.avgG, nn.DataParallel):
            out_state['avgG'] = self.avgG.module.state_dict()
        else:
            out_state['avgG'] = self.avgG.state_dict()
        

        if saveTrainTmp:
            out_state['tmp'] = self.trainTmp

        return out_state

    def save(self, path, saveTrainTmp=False):
        r"""
        Save the model at the given location.

        All parameters included in the self.config class will be saved as well.
        Args:
            - path (string): file where the model should be saved
            - saveTrainTmp (bool): set to True if you want to conserve
                                    the training parameters
        """
        torch.save(self.getStateDict(saveTrainTmp=saveTrainTmp), path)

    def update_config(self, config):
        r"""
        Update the object config with new inputs.

        Args:

            config (dict or BaseConfig) : fields of configuration to be updated

            Typically if config = {"learningRate": 0.1} only the learning rate
            will be changed.
        """
        update_config(self.config, config)
        self.updateSolversDevice()

    def load(self,
             path="",
             in_state=None,
             loadG=True,
             loadD=True,
             loadConfig=True,
             finetuning=False):
        r"""
        Load a model saved with the @method save() function

        Args:
            - path (string): file where the model is stored
        """
        in_state = torch.load(path, 
                              map_location="cuda" if self.useGPU else "cpu")
        self.load_state_dict(in_state,
                             loadG=loadG,
                             loadD=loadD,
                             loadConfig=True,
                             finetuning=False)

    def load_state_dict(self,
                        in_state,
                        loadG=True,
                        loadD=True,
                        loadConfig=True,
                        finetuning=False):
        r"""
        Load a model saved with the @method save() function

        Args:
            - in_state (dict): state dict containing the model
        """

        # Step one : load the configuration
        if loadConfig:
            update_config(self.config, in_state['config'])
            self.lossCriterion = getattr(
                base_loss_criterions, self.config.lossCriterion)(self.device)
            self.initializeClassificationCriterion()

        # Re-initialize G and D with the loaded configuration
        buildAvG = True

        if loadG:
            self.netG = self.getNetG()
            if finetuning:
                loadPartOfStateDict(
                    self.netG, in_state['netG'], ["formatLayer"])
                self.getOriginalG().initFormatLayer(self.config.latentVectorDim)
            else:
                # Replace me by a standard loadStateDictCompatibletedict for open-sourcing TODO
                loadStateDictCompatible(self.netG, in_state['netG'])
                if 'avgG' in in_state:
                    print("Average network found !")
                    self.buildAvG()
                    # Replace me by a standard loadStatedict for open-sourcing
                    if isinstance(self.avgG, nn.DataParallel):

                        # loadStateDictCompatible(self.avgG.module, in_state['avgG'])
                        
                        # HACK TO BE ABLE TO LOAD THE MODELS TRAINED SO FAR
                        loadStateDictCompatible(self.avgG.module, in_state['avgG'])
                    else:
                        loadStateDictCompatible(self.avgG, in_state['avgG'])
                    buildAvG = False

        if loadD:

            self.netD = self.getNetD()
            if finetuning:
                loadPartOfStateDict(
                    self.netD, in_state['netD'], ["decisionLayer"])
                self.getOriginalD().initDecisionLayer(
                    self.lossCriterion.sizeDecisionLayer
                    + self.config.categoryVectorDim)
            else:
                # Replace me by a standard loadStatedict for open-sourcing TODO
                loadStateDictCompatible(self.netD, in_state['netD'])

        elif 'tmp' in in_state.keys():
            self.trainTmp = in_state['tmp']
        # Don't forget to reset the machinery !
        self.updateSolversDevice(buildAvG)

    def classificationPenalty(self, outputD, target, weight, backward=True, skipAtts=False):
        r"""
        Compute the classification penalty associated with the current
        output

        Args:
            - outputD (tensor): discriminator's output
            - target (tensor): ground truth labels
            - weight (float): weight to give to this loss
            - backward (bool): do we back-propagate the loss ?

        Returns:
            - outputD (tensor): updated discrimator's output
            - loss (float): value of the classification loss
        """
        if self.ClassificationCriterion is not None:
            loss = weight * \
                self.ClassificationCriterion.getCriterion(outputD, target, skipAtts=skipAtts)
            if torch.is_tensor(loss):
                if backward:
                    loss.backward(retain_graph=True)

                return loss.item()
        return 0

    def countParams(self):
        import json

        netG = self.getOriginalG()
        netD = self.getOriginalD()
        param_count = dict(G=dict(total=0), D=dict(total=0))

        for name, params in netG.named_parameters():
            if params.requires_grad == True:
                n_params = params.numel()
                param_count['G'][name] = n_params
                param_count['G']['total'] += n_params

        for name, params in netD.named_parameters():
            if params.requires_grad == True:
                n_params = params.numel()
                param_count['D'][name] = n_params
                param_count['D']['total'] += n_params
        return json.dumps(param_count, indent=4)

    def register_D_grads(self):
        if self.register_grads:
            if not hasattr(self, 'gradsD'):
                self.gradsD = {}
            for name, params in self.netD.named_parameters():
                if params.requires_grad == True:
                    try:
                        self.gradsD[name] = params.grad.data
                    except Exception as e:
                        if 'fromRGBLayers' in name: continue
                        print(f"Error grads: {name}")
                        print(e)
                        continue

    def register_G_grads(self):
        if self.register_grads:
            if not hasattr(self, 'gradsG'):
                self.gradsG = {}
            for name, params in self.netG.named_parameters():
                if params.requires_grad == True:
                    try:
                        self.gradsG[name] = params.grad.data
                    except Exception as e:
                        if 'toRGBLayers' in name: continue
                        print(f"Error grads: {name}")
                        print(e)
                        continue