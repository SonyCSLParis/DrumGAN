
import ipdb
import librosa

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
                 plot_iter=50,
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
        self.ignore_phase = False
        self.sanity = False
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
        for i, depth in enumerate(self.config.depthOtherScales):
            groups = 1
            if i > len(self.config.depthOtherScales) - 4:
                groups = 1
            gnet.addScale(depth, groups)

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

    def get_noise_fact(self, iter):
        mse = True
        mse_until = 0
        if iter > mse_until:
            mse = False

        noise_fact = 1.

        print(f"mse = {mse}, noise_fact = {noise_fact}")
        return mse, noise_fact

    def test_G(self, z, x, getAvG=False, toCPU=True, **kargs):
        r"""
        Generate some data given the input latent vector.

        Args:
            z (torch.tensor): input latent vector
        """
        z = z.to(self.device)
        x = x.to(self.device)
        if getAvG:
            output = self.avgG(z, x)
        else:
            output = self.netG(z, x)

        output = output[:, :-1]

        if toCPU:
            return output.detach().cpu()
        return output


    def get_lrs_from_file(self):
        with open("lrs.txt", "r") as f:
            lines = f.readlines()

        floats = [float(l) for l in lines]
        res = tuple(floats)

        if len(lines) == 0:
            res = self.lrs
        else:
            self.lrs = res

        return res

    def optimizeD(self, allLosses, iter):
        # if self.lossDslidingAvg < -1000:
        #     self.config.learning_rate[1] = 2e-5
        # else:
        #     self.config.learning_rate[1] = 2e-5

        # print(f"\nSlidingAvg = {self.lossDslidingAvg}")

        _, self.config.learning_rate[1], _, _, _, noise_fact = self.get_lrs_from_file()

        print(f"\nLearningRateD = {self.config.learning_rate[1]}")

        self.optimizerD = self.getOptimizerD()

        batch_size = self.x.size(0)

        inputLatent, _ = self.buildNoiseData(batch_size)

        inputLatent *= noise_fact

        x_fake = self.netG(inputLatent, self.y_generator).detach().float()[:, :-1]

        if self.ignore_phase:
            self.y[:, 1, ...] = 0
            self.x[:, 1, ...] = 0
            x_fake[:, 1, ...] = 0

        self.optimizerD.zero_grad()

        # real data

        if self.sanity:
            true_xy = torch.cat([self.x, self.x], dim=1)
        else:
            true_xy = torch.cat([self.y, self.x, self.x - self.y], dim=1)

        D_real = self.netD(true_xy, False)

        # fake data

        if self.sanity:
            fake_xy = torch.cat([self.x, x_fake], dim=1)
        else:
            fake_xy = torch.cat([self.y_generator, x_fake, x_fake - self.y_generator], dim=1)

        D_fake = self.netD(fake_xy, False)

        # OBJECTIVE FUNCTION FOR TRUE AND FAKE DATA
        lossD = self.lossCriterion.getCriterion(D_real, False)
        allLosses["lossD_real"] = lossD.item()

        lossDFake = self.lossCriterion.getCriterion(D_fake, False)
        allLosses["lossD_fake"] = lossDFake.item()

        lossD = -lossD + lossDFake

        allLosses["Spread_R-F"] = lossD.item()

        self.lossDslidingAvg = \
            self.lossDslidingAvg * 0.5 + allLosses["Spread_R-F"] * 0.5

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
            lossD = lossD + lossEpsilon
            allLosses["lossD_Epsilon"] = lossEpsilon.item()

        lossD.backward()

        finiteCheck(self.netD.parameters())
        self.optimizerD.step()

        # Logs
        lossD = 0
        for key, val in allLosses.items():

            if key.find("lossD") == 0:
                lossD = lossD + val

        allLosses["lossD"] = lossD

        return allLosses

    def optimizeG(self, allLosses, iter):

        self.config.learning_rate[0], _, mse_fact, adv_fact, mask_fact, noise_fact = self.get_lrs_from_file()

        print(f"LearningRateG = {self.config.learning_rate[0]}")

        sig = nn.Sigmoid()
        self.optimizerG = self.getOptimizerG()
        batch_size = self.x.size(0)
        # Update the generator
        self.optimizerG.zero_grad()
        self.optimizerD.zero_grad()

        # #1 Image generation
        inputLatent, _ = self.buildNoiseData(batch_size)

        inputLatent *= noise_fact
        if self.sanity:
            inputLatent = inputLatent * 0 + 1

        x_fake = self.netG(inputLatent, self.y_generator)

        if self.ignore_phase:
            self.y_generator[:, 1, ...] = 0
            x_fake[:, 1, ...] = 0
            self.x[:, 1, ...] = 0


        if iter % self.plot_iter == 0:
            save_spectrogram("plots", f"wav_{iter}.png",
                             self.x.cpu().detach().numpy()[0, :2])
            save_spectrogram("plots", f"gen_{iter}.png",
                             x_fake.cpu().detach().numpy()[0, :2])

            inputLatent2, _ = self.buildNoiseData(batch_size)
            inputLatent2 *= noise_fact
            with torch.no_grad():
                x_fake2 = self.netG(inputLatent2, self.y_generator)
                save_spectrogram("plots", f"gen2_{iter}.png",
                                 x_fake2.cpu().detach().numpy()[0, :2])

            save_spectrogram("plots", f"mp3_{iter}.png",
                             self.y_generator.cpu().detach().numpy()[0, :2])

        # #2 Status evaluation
        if self.sanity:
            fake_xy = torch.cat([self.x, x_fake[:, :-1]], dim=1)
        else:
            fake_xy = torch.cat([self.y_generator, x_fake[:, :-1], x_fake[:, :-1] - self.y_generator], dim=1)

        D_fake = self.netD(fake_xy, False)

        # #3 GAN criterion
        lossGFake = self.lossCriterion.getCriterion(D_fake, False)
        #lossGFake = -100 * lossGFake / abs(abs(allLosses["Spread_R-F"]) - 100)
        lossGFake = -lossGFake

        MASK = sig(x_fake[:, -1:]) * 0 + 1

        lossMSE = (((x_fake[:, :-1] - self.x) * MASK) ** 2).mean()

        #lossGFake = lossMSE * 0

        allLosses["lossG_fake"] = lossGFake.item()

        lossMASK = MASK.mean()

        lossGFake = lossGFake * adv_fact + lossMSE * mse_fact - lossMASK * mask_fact


        allLosses['mse_loss'] = lossMSE.item()

        print(f"MSE={lossMSE.item()}")
        print(f"Mask={lossMASK.item()}")

        lossGFake.backward()
        finiteCheck(self.getOriginalG().parameters())
        self.register_G_grads()
        self.optimizerG.step()

        lossG = lossGFake * 0
        for key, val in allLosses.items():
            if key.find("lossG") == 0:
                lossG = lossG + val

        allLosses["lossG"] = lossG.item()

        # Update the moving average if relevant
        if isinstance(self.avgG, nn.DataParallel):
            avgGparams = self.avgG.module.parameters()
        else:
            avgGparams = self.avgG.parameters()       
        
        for p, avg_p in zip(self.getOriginalG().parameters(),
                            avgGparams):
            avg_p.mul_(0.999).add_(0.001, p.data)

        return allLosses

    def optimizeParameters(self, x, y, y_generator=None, iter=None):
        allLosses = {}
        # Retrieve the input data
        self.x = x.to(self.device).float()
        self.y = y.to(self.device).float()
        if y_generator is None:
            self.y_generator = self.y
        else:
            self.y_generator = y_generator.to(self.device).float()

        allLosses = self.optimizeD(allLosses, iter)
        allLosses = self.optimizeG(allLosses, iter)

        return allLosses
