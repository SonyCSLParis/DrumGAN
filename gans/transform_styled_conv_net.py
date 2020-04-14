import torch
import torch.nn as nn
import torch.nn.functional as F

import ipdb

from .custom_layers import EqualizedConv2d, EqualizedLinear, \
    NormalizationLayer, Upscale2d, AudioNorm, \
    StyledConv2DBlock, Conv2DBlock, ConstantInput2D, GANsynthInitFormatLayer, \
    StyledConv2DBlockShallow
from utils.utils import num_flat_features
from .mini_batch_stddev_module import miniBatchStdDev
from .progressive_conv_net import DNet
from .styled_progressive_conv_net import StyledGNet
from .utils import scale_interp
import random


def add_grad_map(x):
    # Adds a top-down gradient map (overwrites first map of x)
    x = x.clone()
    gradv = torch.linspace(0, 1, x.shape[2])
    gradh = torch.linspace(0, 1, x.shape[3])

    nr_reps = 4
    rep_length = x.shape[2] // nr_reps
    gradv_rep0 = torch.linspace(0, 1, rep_length).repeat(nr_reps)

    nr_reps = 8
    rep_length = x.shape[2] // nr_reps
    gradv_rep1 = torch.linspace(0, 1, rep_length).repeat(nr_reps)

    if torch.cuda.is_available():
        gradv = gradv.cuda()
        gradh = gradh.cuda()
        gradv_rep0 = gradv_rep0.cuda()
        gradv_rep1 = gradv_rep1.cuda()

    x[:, 0:1, :, :] = gradv[None, None, :, None]
    #x[:, 1:2, :, :] = gradv_rep0[None, None, :, None]
    #x[:, 2:3, :, :] = gradv_rep1[None, None, :, None]
    x[:, 3:4, :, :] = gradh[None, None, None, :]
    return x


def add_input(x, inp):
    x = x.clone()
    x[:, 4:6, ...] = inp
    return x


def shift_maps(x):
    size = x.shape[2]
    nr_maps = 16
    step = size // nr_maps
    for i, shift in enumerate(range(step, size - step, step)):
        x[:, -(i + 1), :(size - shift), :] = x[:, -(i + 1), shift:, :]
        x[:, -(i + 1), (size - shift):, :] = 0

    return x


class TStyledGNet(StyledGNet):
    """

    """
    def __init__(self, **kargs):
        r"""
        Build a generator for a progressive GAN model

        Args:

            - dimLatent (int): dimension of the latent vector
            - depthScale0 (int): depth of the lowest resolution scales
            - initBiasToZero (bool): should we set the bias to zero when a
                                    new scale is added
            - leakyReluLeak (float): leakyness of the leaky relu activation
                                    function
            - normalization (bool): normalize the input latent vector
            - generationActivation (function): activation function of the last
                                               layer (RGB layer). If None, then
                                               the identity is used
            - dimOutput (int): dimension of the output image. 3 -> RGB, 1 ->
                               grey levels
            - equalizedlR (bool): set to true to initiualize the layers with
                                  N(0,1) and apply He's constant at runtime

        """
        self.add_gradient_map = True
        self.uNet = True
        StyledGNet.__init__(self, **kargs)

    def initFormatLayer(self):
        r"""
        The format layer represents the first weights applied to the latent
        vector. It converts a 1xdimLatent input into a 4 x 4 x scalesDepth[0]
        layer.
        """

        self.formatLayer = StyledConv2DBlock(
                                            in_channel=self.dimOutput,
                                            out_channel=self.scalesDepth[0],
                                            kernel_size=self.kernelSize, 
                                            padding=self.padding,
                                            style_dim=self.dimLatent,
                                            init_size=self.sizeScale0,
                                            transposed=self.transposed,
                                            noise_injection=self.noise_injection)

    def initStyleBlock(self):
        layers = [AudioNorm()]
        for i in range(self.n_mlp):
            layers.append(EqualizedLinear(self.dimLatent, self.dimLatent))

            layers.append(nn.LeakyReLU(0.2))

        self.style = nn.Sequential(*layers)

    def initScale0Layer(self):
        self.scaleLayers.append(StyledConv2DBlock(
                                        in_channel=self.scalesDepth[0],
                                        out_channel=self.scalesDepth[0],
                                        kernel_size=self.kernelSize, 
                                        padding=self.padding,
                                        style_dim=self.dimLatent,
                                        init_size=self.sizeScale0,
                                        transposed=self.transposed,
                                        noise_injection=self.noise_injection))

        self.toRGBLayers.append(EqualizedConv2d(self.scalesDepth[0], 
                                                self.dimOutput, 1,
                                                transposed=self.transposed,
                                                equalized=self.equalizedlR,
                                                initBiasToZero=self.initBiasToZero))

    def addScale(self, depthNewScale, groups=1):
        r"""
        Add a new scale to the model. Increasing the output resolution by
        a factor 2

        Args:
            - depthNewScale (int): depth of each conv layer of the new scale
        """
        if type(depthNewScale) is list:
            depthNewScale = depthNewScale[0]
        depthLastScale = self.scalesDepth[-1]
        self.scalesDepth.append(depthNewScale) 
        self.scaleLayers.append(StyledConv2DBlock(
                                                in_channel=depthLastScale,
                                                out_channel=depthNewScale,
                                                kernel_size=self.kernelSize, 
                                                padding=self.padding,
                                                style_dim=self.dimLatent,
                                                init_size=self.sizeScale0,
                                                transposed=self.transposed,
                                                noise_injection=self.noise_injection,
                                                groups=groups))


        self.toRGBLayers.append(EqualizedConv2d(depthNewScale,
                                                self.dimOutput + 1,
                                                1, 
                                                transposed=self.transposed,
                                                equalized=self.equalizedlR,
                                                initBiasToZero=self.initBiasToZero))

    def forward(self,
                input_z,
                input_x,
                noise=None, 
                scale=0, 
                mean_style=None, 
                style_weight=0):
        # Generator

        x_copy = input_x.clone()
        step = len(self.toRGBLayers) - 1
        style = self.style(input_z)
        batch_size = input_z.size(0)
        noise_dim = (batch_size,
                     1,
                     self.outputSizes[-1][0],
                     self.outputSizes[-1][1])
        if noise is None:
            noise = []
            for i in range(step + 1):
                noise.append(torch.randn(noise_dim, device=input_z.device))
        
        out = self.formatLayer(input_x,
                               style=style,
                               noise=torch.randn(noise_dim, device=input_z.device))
        out = add_grad_map(out)
        #out = shift_maps(out)

        padding = 3

        outs = []

        for i, (conv, to_rgb) in enumerate(zip(self.scaleLayers, self.toRGBLayers)):
            out = F.pad(out, [padding] * 4, mode="reflect")
            out = conv(out, style, noise[i])
            out = F.pad(out, [-padding] * 4, mode="reflect")

            out = scale_interp(out, size=self.outputSizes[i])
            out = add_grad_map(out)
            #out = add_input(out, scale_interp(x_copy, size=self.outputSizes[i]))

            if self.uNet and i > self.nScales // 2: # and i < self.nScales - 1:
                try:
                    out = out + outs[self.nScales-i-1]
                except RuntimeError:
                    # If the UNet is not symmetric (additional output layers)
                    pass
            elif self.uNet:
                outs.append(out)

            # if i < len(self.scaleLayers) - 1:
            #     out = shift_maps(out)

        return to_rgb(out)

    def mean_style(self, input):
        style = self.style(input).mean(0, keepdim=True)

        return style


class TStyledDNet(DNet):
    def __init__(self, **args):
        args['dimInput'] *= 2
        args['miniBatchNormalization'] = False
        self.uNet = False
        DNet.__init__(self, **args)

    def initScale0Layer(self):
        # Minibatch standard deviation
        dimEntryScale0 = self.depthScale0
        self.fromRGBLayers.append(EqualizedConv2d(self.dimInput + 2, self.depthScale0, 1,
                                                  equalized=self.equalizedlR,
                                                  initBiasToZero=self.initBiasToZero))
        self.groupScaleZero.append(EqualizedConv2d(dimEntryScale0, self.depthScale0,
                                                   self.kernelSize, padding=self.padding,
                                                   equalized=self.equalizedlR,
                                                   initBiasToZero=self.initBiasToZero))

        self.groupScaleZero.append(EqualizedLinear(32768, # here we have to multiply times the initial size (8 for generating 4096 in 9 scales)
                                                   512,
                                                   equalized=self.equalizedlR,
                                                   initBiasToZero=self.initBiasToZero))

        self.groupScaleZero.append(EqualizedLinear(512,
                                                   # here we have to multiply times the initial size (8 for generating 4096 in 9 scales)
                                                   256,
                                                   equalized=self.equalizedlR,
                                                   initBiasToZero=self.initBiasToZero))

        self.groupScaleZero.append(EqualizedLinear(256,
                                                   # here we have to multiply times the initial size (8 for generating 4096 in 9 scales)
                                                   64,
                                                   equalized=self.equalizedlR,
                                                   initBiasToZero=self.initBiasToZero))

    def initDecisionLayer(self, sizeDecisionLayer):
        self.decisionLayer = EqualizedLinear(64,
                                             sizeDecisionLayer,
                                             equalized=self.equalizedlR,
                                             initBiasToZero=self.initBiasToZero)

    def forward(self, x, getFeature = False):
        pool = torch.nn.MaxPool2d(3, stride=2, padding=1)

        # From RGB layer
        selu = nn.SELU()
        x = self.leakyRelu(self.fromRGBLayers[-1](x))

        #x = shift_maps(x)
        #x = add_grad_map(x)

        # Caution: we must explore the layers group in reverse order !
        # Explore all scales before 0

        shift = len(self.fromRGBLayers) - 2
        nScales = len(self.fromRGBLayers) - 1

        padding = 4

        outs = []
        if self.uNet:
            outs.append(x)

        for i, groupLayer in enumerate(reversed(self.scaleLayers)):
            if i > 4:

                x = F.pad(x, [padding] * 4, mode="reflect")

                for layer in groupLayer:
                    x = self.leakyRelu(layer(x))
                #x = scale_interp(x, size=self.inputSizes[shift], mode="bilinear")

                x = F.pad(x, [-padding] * 4)

                if i != 6 and i != 8:
                    x = pool(x)
                #x = add_grad_map(x)

                if self.uNet and i >= nScales // 2:
                    try:
                        x = x + outs[nScales-i-1]
                    except RuntimeError:
                        # If the UNet is not symmetric (additional output layers)
                        pass
                elif self.uNet:
                    outs.append(x)

                shift -= 1

        # Now the scale 0
        # Minibatch standard deviation
        if self.miniBatchNormalization:
            x = miniBatchStdDev(x)

        #print(self.groupScaleZero[0])
        x = self.leakyRelu(self.groupScaleZero[0](x))
        x = x.view(-1, num_flat_features(x))

        for i in range(1, len(self.groupScaleZero)):
            x_lin = self.groupScaleZero[i](x)
            x = self.leakyRelu(x_lin)

        out = self.decisionLayer(x)

        if not getFeature:
            return out

        return out, x_lin