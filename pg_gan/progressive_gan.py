import torch.optim as optim
import torch.nn as nn

import ipdb

from .base_GAN import BaseGAN
from utils.config import BaseConfig
from .progressive_conv_net import GNet, DNet

from torch.nn import DataParallel
from .gradient_losses import WGANGPGradientPenalty
from utils.utils import loadPartOfStateDict, finiteCheck, \
    loadStateDictCompatible, GPU_is_available
from torch import randperm

import numpy as np


class ProgressiveGAN(BaseGAN):
    r"""
    Implementation of NVIDIA's progressive GAN.
    """

    def __init__(self,
                 depthScales,
                 dimLatentVector=512,
                 initBiasToZero=True,
                 leakyness=0.2,
                 perChannelNormalization=True,
                 miniBatchStdDev=False,
                 equalizedlR=True,
                 sizeScale0=[8, 8],
                 transposed=False,
                 **kwargs):
        r"""
        Args:

        Specific Arguments:
            - depthScale0 (int)
            - initBiasToZero (bool): should layer's bias be initialized to
                                     zero ?
            - leakyness (float): negative slope of the leakyRelU activation
                                 function
            - perChannelNormalization (bool): do we normalize the output of each
                                              convolutional layer ?
            - miniBatchStdDev (bool): mini batch regularization for the
                                      discriminator
            - equalizedlR (bool): if True, forces the optimizer to see weights
                                  in range (-1, 1)

        """
        if not hasattr(self, 'config'):
            self.config = BaseConfig()

        # HACK to have different depth scales in G and D
        # We assume that depthScales[0][0] is G and [0][1] D
        if type(depthScales[0]) is list:
            self.config.depthScale0 = depthScales[0][0]
            self.config.depthScale0D = depthScales[0][1]
        else:
            self.config.depthScale0 = depthScales[0]
            self.config.depthScale0D = depthScales[0]

        self.config.initBiasToZero = initBiasToZero
        self.config.leakyReluLeak = leakyness
        self.config.depthOtherScales = []
        self.config.perChannelNormalization = perChannelNormalization
        self.config.alpha = 0
        self.config.miniBatchStdDev = miniBatchStdDev
        self.config.equalizedlR = equalizedlR
        self.config.sizeScale0 = sizeScale0
        self.config.nScales = len(depthScales)
        self.config.output_shape = kwargs.get('output_shape')
        self.config.scaleSizes = self.initScaleShapes(kwargs.get('downSamplingFactor'))
        self.config.transposed = transposed
        BaseGAN.__init__(self, dimLatentVector, **kwargs)

    def initScaleShapes(self, downSamplingFactor):
        h_size = self.config.output_shape[-1]
        w_size = self.config.output_shape[-2]
        return [(int(np.ceil(w_size / df[0])), int(np.ceil(h_size / df[1]))) for df in downSamplingFactor]

    def getNetG(self):
        print("PGAN: Building Generator")
        gnet = GNet(dimLatent=self.config.latentVectorDim,
                          depthScale0=self.config.depthScale0,
                          initBiasToZero=self.config.initBiasToZero,
                          leakyReluLeak=self.config.leakyReluLeak,
                          normalization=self.config.perChannelNormalization,
                          generationActivation=self.lossCriterion.generationActivation,
                          dimOutput=self.config.dimOutput,
                          equalizedlR=self.config.equalizedlR,
                          sizeScale0=self.config.sizeScale0,
                          transposed=self.config.transposed,
                          scaleSizes=self.config.scaleSizes,
                          formatLayerType=self.config.formatLayerType)

        # Add scales if necessary
        for depth in self.config.depthOtherScales:
            gnet.addScale(depth)

        # If new scales are added, give the generator a blending layer
        if self.config.depthOtherScales:
            gnet.setNewAlpha(self.config.alpha)

        return gnet

    def getNetD(self):
        dnet = DNet(self.config.depthScale0D,
                    initBiasToZero=self.config.initBiasToZero,
                    leakyReluLeak=self.config.leakyReluLeak,
                    sizeDecisionLayer=self.lossCriterion.sizeDecisionLayer +
                    self.config.categoryVectorDim,
                    miniBatchNormalization=self.config.miniBatchStdDev,
                    dimInput=self.config.dimOutput,
                    equalizedlR=self.config.equalizedlR,
                    sizeScale0=self.config.sizeScale0,
                    inputSizes=self.config.scaleSizes)
        # Add scales if necessary
        for depth in self.config.depthOtherScales:
            dnet.addScale(depth)

        # If new scales are added, give the discriminator a blending layer
        if self.config.depthOtherScales:
            dnet.setNewAlpha(self.config.alpha)

        return dnet

    def getOptimizerD(self):

        self.config.learningRateD = self.config.learningRate
        if type(self.config.learningRate) is list:
            self.config.learningRateD = self.config.learningRate[1]
        return optim.Adam(filter(lambda p: p.requires_grad, self.netD.parameters()),
                          betas=[0, 0.99], lr=self.config.learningRateD)

    def getOptimizerG(self):
        self.config.learningRateG = self.config.learningRate
        if type(self.config.learningRate) is list:
            self.config.learningRateG = self.config.learningRate[0]
        return optim.Adam(filter(lambda p: p.requires_grad, self.netG.parameters()),
                          betas=[0, 0.99], lr=self.config.learningRateG)

    def addScale(self, depthNewScale):
        r"""
        Add a new scale to the model. The output resolution becomes twice
        bigger.
        """
        self.netG = self.getOriginalG()
        self.netD = self.getOriginalD()

        # Hack to allow different depthScales in G and D
        if type(depthNewScale) is list:
            self.netG.addScale(depthNewScale[0])
            self.netD.addScale(depthNewScale[1])
        else:
            self.netG.addScale(depthNewScale)
            self.netD.addScale(depthNewScale)

        self.config.depthOtherScales.append(depthNewScale)
        self.updateSolversDevice()

    def updateAlpha(self, newAlpha):
        r"""
        Update the blending factor alpha.

        Args:
            - alpha (float): blending factor (in [0,1]). 0 means only the
                             highest resolution in considered (no blend), 1
                             means the highest resolution is fully discarded.
        """
        self.getOriginalG().setNewAlpha(newAlpha)
        self.getOriginalD().setNewAlpha(newAlpha)

        if self.avgG:
            try:
                self.avgG.module.setNewAlpha(newAlpha)
            except Exception as e:
                self.avgG.setNewAlpha(newAlpha)
        self.config.alpha = newAlpha

    def getSize(self):
        r"""
        Get output image size (W, H)
        """
        return self.getOriginalG().getOutputSize()

    def test(self, input, getAvG=False, toCPU=True, **kargs):
        r"""
        Generate some data given the input latent vector.

        Args:
            input (torch.tensor): input latent vector
        """
        if type(input) == list:
            input = [i.to(self.device) for i in input]
        else:
            input = input.to(self.device)
        if getAvG:
            if toCPU:
                out = self.avgG(input, **kargs)
                if type(out) == list:
                    return [o.cpu() for o in out]
                else:
                    return out.cpu()
            else:
                return self.avgG(input, **kargs)
        elif toCPU:
            out = self.netG(input, **kargs)
            if type(out) == list:
                return [o.detach().cpu() for o in out]
            else:
                return out.detach().cpu()
        else:
            out = self.netG(input, **kargs)
            if type(out) == list:
                return [o.detach().cpu() for o in out]
            else:
                return out.detach().cpu()


    def mix_true_fake_batch(self, true_b, fake_b, fake_ratio):
        assert fake_ratio > 0, "True-Fake split <= 0"
        batch_size = true_b.size(0)
        fake_len = int(fake_ratio * batch_size)
        true_b[-fake_len:] = fake_b[:fake_len]
        return true_b[randperm(batch_size)]

    def optimizeD(self, allLosses):
        batch_size = self.real_input.size(0)

        inputLatent1, targetRandCat1 = self.buildNoiseData(batch_size, self.realLabels, skipAtts=True)
        if getattr(self.config, 'style_mixing', False):
            inputLatent2, targetRandCat2 = self.buildNoiseData(batch_size, self.realLabels, skipAtts=True)
            predFakeG = self.netG([inputLatent1, inputLatent2]).detach()
        else:
            predFakeG = self.netG(inputLatent1).detach()

        self.optimizerD.zero_grad()

        if self.mix_true_fake:
            input_batch = self.mix_true_fake_batch(
                self.real_input, 
                predFakeG, 
                self.true_fake_split)
        # #1 Real data
        predRealD = self.netD(self.real_input, False)
        predFakeD, D_fake_latent = self.netD(predFakeG, True)

        # CLASSIFICATION LOSS
        if self.config.ac_gan:
            # Classification criterion for True and Fake data
            allLosses["lossD_classif"] = \
                self.classificationPenalty(predRealD,
                                           self.realLabels,
                                           self.config.weightConditionD,
                                           backward=True)
                #                             + \
                # self.classificationPenalty(predFakeD,
                #                            self.realLabels,
                #                            self.config.weightConditionD * 0.5,
                #                            backward=True)

        # OBJECTIVE FUNCTION FOR TRUE AND FAKE DATA
        lossD = self.lossCriterion.getCriterion(predRealD, True)
        allLosses["lossD_real"] = lossD.item()

        lossDFake = self.lossCriterion.getCriterion(predFakeD, False)
        allLosses["lossD_fake"] = lossDFake.item()
        lossD += lossDFake

        # #3 WGAN Gradient Penalty loss
        if self.config.lambdaGP > 0:
            allLosses["lossD_GP"], allLosses["lipschitz_norm"] = \
                WGANGPGradientPenalty(input=self.real_input,
                                        fake=predFakeG,
                                        discriminator=self.netD,
                                        weight=self.config.lambdaGP,
                                        backward=True)

        # #4 Epsilon loss
        if self.config.epsilonD > 0:
            lossEpsilon = (predRealD[:, -1] ** 2).sum() * self.config.epsilonD
            lossD += lossEpsilon
            allLosses["lossD_Epsilon"] = lossEpsilon.item()

        # # 5 Logistic gradient loss
        if self.config.logisticGradReal > 0:
            allLosses["lossD_logistic"] = \
                logisticGradientPenalty(self.real_input, self.netD,
                                        self.config.logisticGradReal,
                                        backward=True)
        lossD.backward()

        # self.register_D_grads()
        # finiteCheck(self.netD.module.parameters())
        finiteCheck(self.netD.parameters())
        self.optimizerD.step()

        # Logs
        lossD = 0
        for key, val in allLosses.items():

            if key.find("lossD") == 0:
                lossD += val

        allLosses["lossD"] = lossD

        return allLosses  

    def optimizeG(self, allLosses):
        batch_size = self.real_input.size(0)
        # Update the generator
        self.optimizerG.zero_grad()
        self.optimizerD.zero_grad()

        # #1 Image generation
        inputLatent, targetCatNoise = self.buildNoiseData(batch_size, self.realLabels, skipAtts=True)

        if getattr(self.config, 'style_mixing', False):
            inputLatent2, targetRandCat2 = self.buildNoiseData(batch_size, self.realLabels, skipAtts=True)
            predFakeG = self.netG([inputLatent, inputLatent2])
        else:
            predFakeG = self.netG(inputLatent)

        # #2 Status evaluation
        predFakeD, phiGFake = self.netD(predFakeG, True)

        # #2 Classification criterion
        if self.config.ac_gan:
            G_classif_fake = \
                self.classificationPenalty(predFakeD,
                                           targetCatNoise,
                                           self.config.weightConditionG,
                                           backward=True,
                                           skipAtts=True)
            allLosses["lossG_classif"] = G_classif_fake
        # #3 GAN criterion
        lossGFake = self.lossCriterion.getCriterion(predFakeD, True)
        allLosses["lossG_fake"] = lossGFake.item()
        allLosses["Spread_R-F"] = allLosses["lossD_real"] - allLosses["lossG_fake"]

        # Back-propagate generator losss
        lossGFake.backward()
        finiteCheck(self.getOriginalG().parameters())
        self.register_G_grads()
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

    def optimizeParameters(self, input_batch, inputLabels, fakeLabels=None):
        allLosses = {}
        try:
            allLosses['alpha'] = self.getOriginalD().alpha
        except AttributeError:
            pass

        if fakeLabels is None:
            self.fakeLabels = inputLabels
        else:
            self.fakeLabels = fakeLabels
        # Retrieve the input data
        self.real_input, self.realLabels = input_batch.to(self.device).float(), None
        if self.config.attribKeysOrder is not None:
            self.realLabels = inputLabels.to(self.device)

        n_samples = self.real_input.size()[0]

        allLosses = self.optimizeD(allLosses)
        allLosses = self.optimizeG(allLosses)
        return allLosses


    def optimizeParameters2(self, input_batch, inputLabels=None):
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
        self.real_input, self.realLabels = input_batch.to(self.device).float(), None
        if self.config.attribKeysOrder is not None:
            self.realLabels = inputLabels.to(self.device)

        n_samples = self.real_input.size()[0]

        # Update the discriminator
        self.optimizerD.zero_grad()

        # #1 Real data
        predRealD = self.netD(self.real_input, False)
        # #2 Fake data
        inputLatent, targetRandCat = self.buildNoiseData(n_samples, self.realLabels, skipAtts=True)
        
        if getattr(self.config, 'style_mixing', False):
            inputLatent2, targetRandCat2 = self.buildNoiseData(n_samples, self.realLabels, skipAtts=True)
            predFakeG = self.netG([inputLatent, inputLatent2]).detach()
        else:
            predFakeG = self.netG(inputLatent).detach()

        predFakeD, D_fake_latent = self.netD(predFakeG, True)

        if self.config.ac_gan:
            # Classification criterion for True and Fake data
            allLosses["lossD_classif"] = \
                self.classificationPenalty(predRealD,
                                           self.realLabels,
                                           self.config.weightConditionD * 0.5,
                                           backward=True) + \
                self.classificationPenalty(predFakeD,
                                           self.realLabels,
                                           self.config.weightConditionD * 0.5,
                                           backward=True)

        lossD = self.lossCriterion.getCriterion(predRealD, True)
        allLosses["lossD_real"] = lossD.item()

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
            lossEpsilon = (predRealD[:, -1] ** 2).sum() * self.config.epsilonD
            lossD += lossEpsilon
            allLosses["lossD_Epsilon"] = lossEpsilon.item()


        # # 5 Logistic gradient loss
        if self.config.logisticGradReal > 0:
            allLosses["lossD_logistic"] = \
                logisticGradientPenalty(self.real_input, self.netD,
                                        self.config.logisticGradReal,
                                        backward=True)
        lossD.backward(retain_graph=True)

        self.register_D_grads()
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
        inputLatent, targetCatNoise = self.buildNoiseData(n_samples, self.realLabels, skipAtts=True)

        if getattr(self.config, 'style_mixing', False):
            inputLatent2, targetRandCat2 = self.buildNoiseData(n_samples, self.realLabels, skipAtts=True)
            predFakeG = self.netG([inputLatent, inputLatent2])
        else:
            predFakeG = self.netG(inputLatent)

        # #2 Status evaluation
        predFakeD, phiGFake = self.netD(predFakeG, True)

        # #2 Classification criterion
        if self.config.ac_gan:
            G_classif_fake = \
                self.classificationPenalty(predFakeD,
                                           targetCatNoise,
                                           self.config.weightConditionG,
                                           backward=True,
                                           skipAtts=True)
            allLosses["lossG_classif"] = G_classif_fake
        # #3 GAN criterion
        lossGFake = self.lossCriterion.getCriterion(predFakeD, True)
        allLosses["lossG_fake"] = lossGFake.item()

        allLosses["Spread_R-F"] = allLosses["lossD_real"] + allLosses["lossD_fake"]

        # Back-propagate generator losss
        lossGFake.backward()
        finiteCheck(self.getOriginalG().parameters())
        self.register_G_grads()
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

