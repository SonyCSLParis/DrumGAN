import ipdb

import torch.nn as nn

import torch
from .styled_progressive_conv_net import StyledGNet, StyledDNet
from utils.config import BaseConfig
from .progressive_gan import ProgressiveGAN
from .styled_progressive_conv_net import DNet

class StyleProgressiveGAN(ProgressiveGAN):
    r"""
    Implementation of NVIDIA's progressive GAN.
    """

    def __init__(self, 
                 n_mlp=8,
                 noise_injection=True,
                 style_mixing=True,
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
        ProgressiveGAN.__init__(self, **kwargs)
        

    def getNetG(self):
        gnet = StyledGNet(dimLatent=self.config.latentVectorDim,
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
        dnet = DNet(self.config.depthScale0D,
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
