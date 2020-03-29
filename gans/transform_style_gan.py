import ipdb

import torch.nn as nn

import torch
from .transform_styled_conv_net import TStyledGNet, TStyledDNet
from utils.config import BaseConfig
from .progressive_gan import ProgressiveGAN
from .utils import save_spectrogram

from .gradient_losses import WGANGPGradientPenalty
from utils.utils import loadPartOfStateDict, finiteCheck, \
    loadStateDictCompatible, GPU_is_available

class TStyleGAN(ProgressiveGAN):
    r"""
    Implementation of NVIDIA's progressive GAN.
    """

    def __init__(self, 
                 n_mlp=8,
                 noise_injection=True,
                 style_mixing=True,
                 plot_iter=100,
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
                                  in range (-1, 1
        """

        # SUPER HACK: the number of scales should be in the config file

        if not hasattr(self, 'config'):
            self.config = BaseConfig()
            self.config.noise_injection = noise_injection
            self.config.style_mixing = style_mixing

        self.n_mlp = n_mlp
        self.plot_iter = plot_iter
        self.lossDslidingAvg = -0.
        ProgressiveGAN.__init__(self, **kwargs)
        

    def getNetG(self):
        gnet = TStyledGNet(dimLatent=self.config.latentVectorDim,
                          depthScale0=self.config.depthScale0,
                          initBiasToZero=self.config.initBiasToZero,
                          leakyReluLeak=self.config.leakyReluLeak,
                          normalization=self.config.perChannelNormalization,
                          generationActivation=self.lossCriterion.generationActivation,
                          dimOutput=self.config.dimOutput,
                          equalizedlR=self.config.equalizedlR,
                          sizeScale0=self.config.sizeScale0,
                          outputSizes=self.config.scaleSizes,
                          nScales=self.config.nScales,
                          n_mlp=self.n_mlp,
                          transposed=self.config.transposed,
                          noise_injection=self.config.noise_injection,
                          formatLayerType=self.config.formatLayerType)

        # Add scales if necessary
        for depth in self.config.depthOtherScales:
            gnet.addScale(depth)

        # If new scales are added, give the generator a blending layer
        if self.config.depthOtherScales:
            gnet.setNewAlpha(self.config.alpha)

        return gnet

    def getNetD(self):

        dnet = TStyledDNet(depthScale0=self.config.depthScale0D,
                           initBiasToZero=self.config.initBiasToZero,
                           leakyReluLeak=self.config.leakyReluLeak,
                           sizeDecisionLayer=self.lossCriterion.sizeDecisionLayer +
                           self.config.categoryVectorDim,
                           miniBatchNormalization=self.config.miniBatchStdDev,
                           dimInput=self.config.dimOutput,
                           equalizedlR=self.config.equalizedlR,
                           sizeScale0=self.config.sizeScale0,
                           inputSizes=self.config.scaleSizes,
                           nScales=self.config.nScales)

        # Add scales if necessary
        for depth in self.config.depthOtherScales:
            dnet.addScale(depth)

        # If new scales are added, give the discriminator a blending layer
        if self.config.depthOtherScales:
            dnet.setNewAlpha(self.config.alpha)

        return dnet

    def optimizeD(self, allLosses, iter):
        mse, noise_fact = self.get_noise_fact(iter)

        if self.lossDslidingAvg < -500:
            self.config.learningRate[1] = 0.00003
        else:
            self.config.learningRate[1] = 0.0006

        if mse:
            self.config.learningRate[1] = 0.

        print(f"\nSlidingAvg = {self.lossDslidingAvg}")
        print(f"LearningRateD = {self.config.learningRate[1]}")

        self.optimizerD = self.getOptimizerD()

        batch_size = self.x.size(0)

        inputLatent, _ = self.buildNoiseData(batch_size)
        y_fake = self.netG(inputLatent, self.x_generator).detach().float()

        self.optimizerD.zero_grad()

        # real data
        true_xy = torch.cat([self.x, self.y], dim=1)
        D_real = self.netD(true_xy, False)

        if iter % self.plot_iter == 0:
            save_spectrogram(f"wav_spect_{iter}.png",
                             self.y.cpu().detach().numpy()[0, 0])
            save_spectrogram(f"wav_phase_{iter}.png",
                             self.y.cpu().detach().numpy()[0, 1])

        # fake data
        fake_xy = torch.cat([self.x_generator, y_fake], dim=1)
        D_fake = self.netD(fake_xy, False)

        # OBJECTIVE FUNCTION FOR TRUE AND FAKE DATA
        lossD = self.lossCriterion.getCriterion(D_real, False)
        allLosses["lossD_real"] = lossD.item()

        lossDFake = self.lossCriterion.getCriterion(D_fake, False)
        allLosses["lossD_fake"] = lossDFake.item()
        lossD = -lossD + lossDFake

        lossD *= noise_fact

        allLosses["Spread_R-F"] = lossD.item()

        self.lossDslidingAvg = self.lossDslidingAvg * 0.5 + allLosses[
            "Spread_R-F"] * 0.5

        # #3 WGAN Gradient Penalty loss
        if self.config.lambdaGP > 0:
            allLosses["lossD_GP"], allLosses["lipschitz_norm"] = \
                WGANGPGradientPenalty(input=true_xy,
                                        fake=fake_xy,
                                        discriminator=self.netD,
                                        weight=self.config.lambdaGP,
                                        backward=True)

        # #4 Epsilon loss
        if self.config.epsilonD > 0:
            lossEpsilon = (D_real[:, -1] ** 2).sum() * self.config.epsilonD
            lossD += lossEpsilon
            allLosses["lossD_Epsilon"] = lossEpsilon.item()

        lossD.backward()

        finiteCheck(self.netD.parameters())
        self.optimizerD.step()

        # Logs
        lossD = 0
        for key, val in allLosses.items():

            if key.find("lossD") == 0:
                lossD += val

        allLosses["lossD"] = lossD

        return allLosses

    def get_noise_fact(self, iter):
        mse = True
        mse_until = 0
        if iter > mse_until:
            mse = False

        noise_fact = float(iter - mse_until) #* 1e-4
        noise_fact = min(noise_fact, 1)

        if mse:
            noise_fact = 0.

        print(f"mse = {mse}, noise_fact = {noise_fact}")
        return mse, noise_fact

    def optimizeG(self, allLosses, iter):
        mse, noise_fact = self.get_noise_fact(iter)

        if self.lossDslidingAvg < -500:
            self.config.learningRate[0] = 0.0006
        else:
            self.config.learningRate[0] = 0.0006 #0.00003

        if mse:
            self.config.learningRate[0] = 0.001

        print(f"LearningRateG = {self.config.learningRate[0]}")

        self.optimizerG = self.getOptimizerG()
        batch_size = self.x.size(0)
        # Update the generator
        self.optimizerG.zero_grad()
        self.optimizerD.zero_grad()

        # #1 Image generation
        inputLatent, _ = self.buildNoiseData(batch_size)

        inputLatent *= (noise_fact * 1e-4)

        y_fake = self.netG(inputLatent, self.x_generator)

        if iter % self.plot_iter == 0:
            save_spectrogram(f"gen_spect_{iter}.png", y_fake.cpu().detach().numpy()[0, 0])
            save_spectrogram(f"mp3_spect_{iter}.png", self.x_generator.cpu().detach().numpy()[0, 0])

        # #2 Status evaluation
        fake_xy = torch.cat([self.x_generator, y_fake], dim=1)
        D_fake = self.netD(fake_xy, False)

        # #3 GAN criterion
        lossGFake = self.lossCriterion.getCriterion(D_fake, False)

        lossGFake = -lossGFake * noise_fact

        allLosses["lossG_fake"] = lossGFake.item()

        lossMSE = ((y_fake - self.y) ** 2).mean()

        print(f"Loss MSE = {lossMSE.item()}")

        lossGFake += lossMSE

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

    def optimizeParameters(self, x, y, x_generator=None, iter=None):
        allLosses = {}
        # Retrieve the input data
        self.x = x.to(self.device).float()
        self.y = y.to(self.device).float()
        if x_generator is None:
            self.x_generator = self.x
        else:
            self.x_generator = x_generator.to(self.device).float()

        allLosses = self.optimizeD(allLosses, iter)
        allLosses = self.optimizeG(allLosses, iter)

        return allLosses
