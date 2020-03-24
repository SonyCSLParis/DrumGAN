import torch
import torch.nn as nn
import torch.nn.functional as F

import ipdb

from .custom_layers import EqualizedConv2d, EqualizedLinear,\
    NormalizationLayer, Upscale2d, AudioNorm, \
    StyledConv2DBlock, Conv2DBlock, ConstantInput2D, GANsynthInitFormatLayer
from utils.utils import num_flat_features
from .mini_batch_stddev_module import miniBatchStdDev
from .progressive_conv_net import GNet, DNet
import random


class StyledGNet(nn.Module):
    """

    """
    def __init__(self,
                 dimLatent,
                 depthScale0,
                 n_mlp=8,
                 initBiasToZero=True,
                 leakyReluLeak=0.2,
                 normalization=True,
                 generationActivation=None,
                 dimOutput=1,
                 equalizedlR=True,
                 sizeScale0=16,
                 kernelSize=3,
                 padding=1,
                 outputSizes=0,
                 nScales=1,
                 transposed=False,
                 noise_injection=True,
                 formatLayerType='ConstantInput',
                 **kargs):
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
        super(StyledGNet, self).__init__()
        # super(StyledGNet, self).__init__(dimLatent, depthScale0, **kargs)

        self.dimLatent = dimLatent

        self.equalizedlR = equalizedlR
        self.initBiasToZero = initBiasToZero
        # if type(sizeScale0) == tuple:
        #     self.sizeScale0 = sizeScale0[1]
        #     self.Scale0y = sizeScale0[0]
        # else:
        self.sizeScale0 = sizeScale0
        self.outputSizes = outputSizes
        self.nScales = nScales

        self.n_mlp = n_mlp
        self.kernelSize = kernelSize
        self.padding = padding
        # Initalize the scales
        self.scalesDepth = [depthScale0]
        self.groupScaleZero = nn.ModuleList()
        self.scaleLayers = nn.ModuleList()
        self.toRGBLayers = nn.ModuleList()
        self.dimOutput = dimOutput
        self.depthScale0 = depthScale0
        self.transposed = transposed

        self.noise_injection = noise_injection
        self.formatLayerType = formatLayerType

        self.initFormatLayer()
        self.initStyleBlock()
        self.initScale0Layer()
        # Initalize the upscaling parameters
        # alpha : when a new scale is added to the network, the previous
        # layer is smoothly merged with the output in the first stages of
        # the training
        self.alpha = 0

        # Leaky relu activation
        self.leakyRelu = torch.nn.LeakyReLU(leakyReluLeak)

        # normalization
        self.normalizationLayer = None
        if normalization:
            self.normalizationLayer = NormalizationLayer()

        # Last layer activation function
        self.generationActivation = generationActivation

    def initFormatLayer(self):
        r"""
        The format layer represents the first weights applied to the latent
        vector. It converts a 1xdimLatent input into a 4 x 4 x scalesDepth[0]
        layer.
        """

        # self.dimLatent = dimLatentVector
        if self.formatLayerType in ['default', 'constant_input']:
            self.formatLayer = ConstantInput2D(channel=self.scalesDepth[0], size=self.sizeScale0)
       
        elif self.formatLayerType == 'rand_z':
            self.formatLayer = EqualizedLinear(self.dimLatent,
                                               self.sizeScale0[0] * self.sizeScale0[1] * self.depthScale0, #Change factor for other than 8
                                               equalized=self.equalizedlR,
                                               initBiasToZero=self.initBiasToZero)
                                
        elif self.formatLayerType == 'gansynth':
            self.formatLayer = GANsynthInitFormatLayer(
                    dimLatent=self.dimLatent, 
                    scaleDepth=self.scalesDepth[0], 
                    outputShape=self.sizeScale0, 
                    equalizedlR=self.equalizedlR, 
                    initBiasToZero=self.initBiasToZero
                )

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
    # def getOutputSize(self):
    #     r"""
    #     Get the size of the generated image.
    #     """
    #     # side = 4 * (2**(len(self.toRGBLayers) - 1))
    #     if type(self.sizeScale0) == tuple:
    #         # side1 = int(self.sizeScale0[0] * (2**(len(self.toRGBLayers) - 1)))
    #         # side2 = int(self.sizeScale0[1] * (2**(len(self.toRGBLayers) - 1)))
    #         side1 = int(self.sizeScale0[0] * (2**(len(self.toRGBLayers))))
    #         side2 = int(self.sizeScale0[1] * (2**(len(self.toRGBLayers))))
    #         return self.outputSizes[0]
    #         # return (side1, side2)
    #     else:
    #         side = self.sizeScale0 * (2**(len(self.toRGBLayers) - 1))
    #         # side = self.sizeScale0 * (2**(len(self.toRGBLayers)))
    #         return (side, side)

    def addScale(self, depthNewScale):
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
                                                noise_injection=self.noise_injection))


        self.toRGBLayers.append(EqualizedConv2d(depthNewScale,
                                                self.dimOutput,
                                                1, 
                                                transposed=self.transposed,
                                                equalized=self.equalizedlR,
                                                initBiasToZero=self.initBiasToZero))


    def setNewAlpha(self, alpha):
        r"""
        Update the value of the merging factor alpha

        Args:

            - alpha (float): merging factor, must be in [0, 1]
        """

        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be in [0,1]")

        if not self.toRGBLayers:
            raise AttributeError("Can't set an alpha layer if only the scale 0"
                                 "is defined")

        self.alpha = alpha

    def upsample(self, x, size=0):
        return F.interpolate(x, size=size, mode='nearest')
    
    def forward(self,
                input,
                noise=None, 
                scale=0, 
                alpha=-1, 
                mean_style=None, 
                style_weight=0, 
                mixing_range=(-1, -1),
                test_all_scales=False):

        styles = []
        alpha = 1 - self.alpha
        step = len(self.toRGBLayers) - 1
        output = []

        if type(input) not in (list, tuple):
            input = [input]

        for i in input:
            styles.append(self.style(i))

        batch = input[0].shape[0]

        if noise is None:
            noise = []
            for i in range(step + 1):
                noise_dim = (batch, 
                             1, 
                             self.outputSizes[i][0], 
                             self.outputSizes[i][1])
                noise.append(torch.randn(noise_dim, device=input[0].device))
        
        if mean_style is not None:
            styles_norm = []
            for style in styles:
                styles_norm.append(mean_style + style_weight * (style - mean_style))
            styles = styles_norm
        
        # out = noise[0]
        out = self.formatLayer(input[0])
        out = out.view(input[0].size(0), self.depthScale0, self.sizeScale0[0], self.sizeScale0[1])

        if len(styles) < 2 or step == 0:
            inject_index = [len(self.scaleLayers) + 1]
        else:
            inject_index = random.sample(list(range(step)), len(styles) - 1)

        crossover = 0

        for i, (conv, to_rgb) in enumerate(zip(self.scaleLayers, self.toRGBLayers)):
            if mixing_range == (-1, -1):
                if crossover < len(inject_index) and i > inject_index[crossover]:
                    crossover = min(crossover + 1, len(styles))

                style_step = styles[crossover]

            else:
                if mixing_range[0] <= i <= mixing_range[1]:
                    style_step = styles[1]
                else:
                    style_step = styles[0]


            if i > 0 and step > 0:
                upsample = self.upsample(out, self.outputSizes[i])

                out = conv(upsample, style_step, noise[i])

            else:
                out = conv(out, style_step, noise[i])

            if test_all_scales and i != step:
                output.append(to_rgb(out))

            if i == step:
                out = to_rgb(out)

                if i > 0 and 0 <= alpha < 1:
                    skip_rgb = self.toRGBLayers[i - 1](upsample)
                    # skip_rgb = self.toRGBLayers[-1](upsample)
                    out = (1 - alpha) * skip_rgb + alpha * out
                break

        if test_all_scales:
            output.append(out)
            return output
        else:
            return out

    def mean_style(self, input):
        style = self.style(input).mean(0, keepdim=True)

        return style


class StyledDNet(nn.Module):
    def __init__(self, 
                 depthScale0,
                 initBiasToZero=True,
                 leakyReluLeak=0.2,
                 sizeDecisionLayer=1,
                 miniBatchNormalization=False,
                 dimInput=1,
                 equalizedlR=True,
                 sizeScale0=16,
                 kernelSize=3,
                 padding=1):
        
        super(StyledDNet, self).__init__()
        if type(sizeScale0) == tuple:
            self.sizeScale0 = sizeScale0[1]
            self.Scale0y = sizeScale0[0]

        else:
            self.sizeScale0 = sizeScale0
        
        self.kernelSize = kernelSize
        self.padding = padding

        # Initialization paramneters
        self.initBiasToZero = initBiasToZero
        self.equalizedlR = equalizedlR
        self.dimInput = dimInput
        self.miniBatchNormalization = miniBatchNormalization

        # Initalize the scales
        self.scalesDepth = [depthScale0]
        self.scaleLayers = nn.ModuleList()
        self.fromRGBLayers = nn.ModuleList()
        self.mergeLayers = nn.ModuleList()
        self.initial_size = depthScale0
        self.groupScaleZero = nn.ModuleList()

        # Initialize the last layer
        self.initDecisionLayer(sizeDecisionLayer)

        # Layer 0
        self.initScale0Layer()

        # Initalize the upscaling parameters
        self.alpha = 0

        # Leaky relu activation
        self.leakyRelu = torch.nn.LeakyReLU(leakyReluLeak)
        self.n_layer = len(self.fromRGBLayers)

    def initDecisionLayer(self, sizeDecisionLayer):
        self.decisionLayer = EqualizedLinear(nChannelsPrevious=self.sizeScale0,
                                         nChannels=sizeDecisionLayer)

    def initScale0Layer(self):

        # Minibatch standard deviation
        dimEntryScale0 = self.scalesDepth[0]
        if self.miniBatchNormalization:
            dimEntryScale0 += 1
        if self.dimInput == 1:
            self.groupScaleZero.append(Conv1DBlock(dimEntryScale0, 
                                                 self.scalesDepth[0],
                                                 self.kernelSize,
                                                 padding=self.padding,
                                                 kernel_size2=1,
                                                 padding2=0))      
            self.fromRGBLayers.append(EqualizedConv1d(self.dimInput,
                                                 self.scalesDepth[0], 
                                                 1))
        
        elif self.dimInput > 1:
            self.groupScaleZero.append(Conv2DBlock(dimEntryScale0, 
                                                 self.scalesDepth[0],
                                                 self.kernelSize,
                                                 padding=self.padding,
                                                 kernel_size2=1,
                                                 padding2=0))      
            self.fromRGBLayers.append(EqualizedConv2d(self.dimInput,
                                                 self.scalesDepth[0], 
                                                 1))
        
        self.groupScaleZero.append(EqualizedLinear(self.scalesDepth[0] * \
                                                   self.sizeScale0, # here we have to multiply times the initial size (8 for generating 4096 in 9 scales)
                                                   self.scalesDepth[0]))
            
    def addScale(self, depthNewScale):

        depthLastScale = self.scalesDepth[-1]
        self.scalesDepth.append(depthNewScale)
        if self.dimInput == 1:
            self.scaleLayers.append(Conv1DBlock(depthNewScale,
                                                depthLastScale,
                                                self.kernelSize,
                                                padding=self.padding))

            self.fromRGBLayers.append(EqualizedConv1d(self.dimInput,
                                                depthNewScale,
                                                1))
        elif self.dimInput > 1:
            self.scaleLayers.append(Conv2DBlock(depthNewScale,
                                                depthLastScale,
                                                self.kernelSize,
                                                padding=self.padding))

            self.fromRGBLayers.append(EqualizedConv2d(self.dimInput,
                                                depthNewScale,
                                                1))
    def downsample(self, x):
        if self.dimInput == 1:
            return F.interpolate(x, scale_factor=0.5, mode='linear', align_corners=False)
        elif self.dimInput > 1:
            return F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)

    def setNewAlpha(self, alpha):
        r"""
        Update the value of the merging factor alpha

        Args:

            - alpha (float): merging factor, must be in [0, 1]
        """

        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be in [0,1]")

        if not self.fromRGBLayers:
            raise AttributeError("Can't set an alpha layer if only the scale 0"
                                 "is defined")

        self.alpha = alpha

    def forward2(self, input, getFeature = False):
        alpha = 1 - self.alpha
        scale = len(self.scaleLayers)
        self.n_layer = len(self.scaleLayers)

        for i in range(scale, -1, -1):


            # index = self.n_layer - i - 1
            index = self.n_layer - i - 1

            if i == scale:
                out = self.fromRGBLayers[index](input)
                # out = self.fromRGBLayers[-2](input)

            if i == 0:
                # This block adds a variance feature-map computed from across all the channel dimensions, 
                # it helps the discriminator to spot lack of variance in the batch and speed up convergence
                # read paper at INCREASING VARIATION USING MINIBATCH STANDARD DEVIATION
                out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)
                mean_std = out_std.mean()

                mean_std = mean_std.expand(out.size(0), 1, out.size(2))
                # mean_std = mean_std.expand(out.size(0), 1, 4, 4)
                out = torch.cat([out, mean_std], 1)
                out = self.groupScaleZero[0](out)
            else:

                out = self.scaleLayers[index](out)
            if i > 0:

                # out = F.avg_pool2d(out, 2)
                out = self.downsample(out)


                if i == scale and 0 <= alpha < 1:
                    # skip_rgb = F.avg_pool2d(input, 2)

                    skip_rgb = self.downsample(input)


                    # skip_rgb = self.fromRGBLayers[index + 1](skip_rgb)
                    skip_rgb = self.fromRGBLayers[index - 1](skip_rgb)

                    # Combine output from previous layer + current output weighted by alpha
                    out = (1 - alpha) * skip_rgb + alpha * out


        # squeeze --> Returns a tensor with all the dimensions of input of size 1 removed.
        out = out.squeeze(2)
        # out.view(-1, num_flat_features(out))
        # out = self.leakyRelu(self.groupScaleZero[1](out))
        # Performs a linear classifier (Warssestain GANs always have a linear clf out, NEVER SoftMax)
        
        

        out = self.decisionLayer(out)
        if not getFeature:
            return out

        return out, input
 
    def forward(self, x, getFeature = False):
        # Alpha blending
        if self.alpha > 0 and len(self.fromRGBLayers) > 1:
            # y = F.avg_pool2d(x, (2, 2))
            y = self.downsample(x)
            
            # y = self.leakyRelu(self.fromRGBLayers[-2](y))
            y = self.fromRGBLayers[-2](y)

        # From RGB layer
        # x = self.leakyRelu(self.fromRGBLayers[-1](x))
        x = self.fromRGBLayers[-1](x)

        # Caution: we must explore the layers group in reverse order !
        # Explore all scales before 0
        mergeLayer = self.alpha > 0 and len(self.scaleLayers) > 1
        for groupLayer in reversed(self.scaleLayers):

            x = groupLayer(x)
            x = self.downsample(x)
            if mergeLayer:
                mergeLayer = False
                x = self.alpha * y + (1-self.alpha) * x
        # Now the scale 0

        # Minibatch standard deviation
        if self.miniBatchNormalization:
            x = miniBatchStdDev(x)
            # out_std = torch.sqrt(x.var(0, unbiased=False) + 1e-8)
            # mean_std = out_std.mean()
            # mean_std = mean_std.expand(x.size(0), 1, x.size(2))
            # x = torch.cat([x, mean_std], dim=1)

        x = self.groupScaleZero[0](x)
        out = self.decisionLayer(x)
        if not getFeature:
            return out

        return out, x