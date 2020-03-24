import torch
import torch.nn as nn
import torch.nn.functional as F

from .custom_layers import EqualizedConv1d, EqualizedConv2d, EqualizedLinear,\
    NormalizationLayer, Upscale2d, GANsynthInitFormatLayer
from utils.utils import num_flat_features
from.mini_batch_stddev_module import miniBatchStdDev


class GNet(nn.Module):

    def __init__(self,
                 dimLatent,
                 depthScale0,
                 scaleSizes,
                 initBiasToZero=True,
                 leakyReluLeak=0.2,
                 normalization=True,
                 generationActivation=None,
                 dimOutput=1,
                 equalizedlR=True,
                 sizeScale0=16,
                 outputSize=0,
                 transposed=False,
                 nScales=1,
                 formatLayerType='default'):
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
        super(GNet, self).__init__()

        self.formatLayerType = formatLayerType

        self.equalizedlR = equalizedlR
        self.initBiasToZero = initBiasToZero

        self.sizeScale0 = sizeScale0
        self.outputSize = outputSize
        self.nScales = nScales
        self.transposed = transposed


        print()
        print(f'Size scale 0: {self.sizeScale0}')
        print()
        # Initalize the scales
        self.scalesDepth = [depthScale0]
        self.scaleLayers = nn.ModuleList()
        self.toRGBLayers = nn.ModuleList()

        # Leaky relu activation
        self.leakyRelu = torch.nn.LeakyReLU(leakyReluLeak)
        # Convolution kernels
        self.kernelSize = 3
        self.padding = 1
        # normalization
        self.normalizationLayer = None
        if normalization:
            self.normalizationLayer = NormalizationLayer()

        # Initialize the scale 0
        self.dimOutput = dimOutput
        self.initFormatLayer(dimLatent)
        self.initScale0Layer()

        # Initalize the upscaling parameters
        # alpha : when a new scale is added to the network, the previous
        # layer is smoothly merged with the output in the first stages of
        # the training
        self.alpha = 0

        # Last layer activation function
        self.generationActivation = generationActivation
        self.depthScale0 = depthScale0


        self.scaleSizes = scaleSizes
        


    def initFormatLayer(self, dimLatentVector):
        r"""
        The format layer represents the first weights applied to the latent
        vector. It converts a 1xdimLatent input into a 4 x 4 x scalesDepth[0]
        layer.
        """
        self.dimLatent = dimLatentVector
        if self.formatLayerType in ['default', 'rand_z']:
            self.formatLayer = EqualizedLinear(self.dimLatent,
                                               self.sizeScale0[0] * self.sizeScale0[1] * self.scalesDepth[0], #Change factor for other than 8
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

    def initScale0Layer(self):
        self.groupScale0 = nn.ModuleList()

        self.groupScale0.append(EqualizedConv2d(self.scalesDepth[0],
                                                self.scalesDepth[0],
                                                self.kernelSize,
                                                equalized=self.equalizedlR,
                                                transposed=self.transposed,
                                                initBiasToZero=self.initBiasToZero,
                                                padding=1))

        self.toRGBLayers.append(EqualizedConv2d(self.scalesDepth[0],
                                                self.dimOutput, 1,
                                                transposed=self.transposed,
                                                equalized=self.equalizedlR,
                                                initBiasToZero=self.initBiasToZero))  
        

    def getOutputSize(self):
        r"""
        Get the size of the generated image.
        """
        if type(self.sizeScale0) == tuple:
            side1 = int(self.sizeScale0[0] * (2**(len(self.toRGBLayers))))
            side2 = int(self.sizeScale0[1] * (2**(len(self.toRGBLayers))))
            return (side1, side2)
        else:
            side = self.sizeScale0 * (2**(len(self.toRGBLayers)))
            # side = self.sizeScale0 * (2**(len(self.toRGBLayers) - 1))
            return (side, side)

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

        self.scaleLayers.append(nn.ModuleList())

        self.scaleLayers[-1].append(EqualizedConv2d(depthLastScale, depthNewScale,
                                                    self.kernelSize, padding=self.padding,
                                                    equalized=self.equalizedlR,
                                                    transposed=self.transposed,
                                                    initBiasToZero=self.initBiasToZero))
        
        self.scaleLayers[-1].append(EqualizedConv2d(depthNewScale, depthNewScale,
                                                    self.kernelSize, padding=self.padding,
                                                    equalized=self.equalizedlR,
                                                    transposed=self.transposed,
                                                    initBiasToZero=self.initBiasToZero))

        self.toRGBLayers.append(EqualizedConv2d(depthNewScale,
                                                self.dimOutput,
                                                1, transposed=self.transposed,
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

    def upscale(self, x, size):
        return F.interpolate(x, size=size, mode='nearest')

    def tensor_view(self, x):
        return x.view(x.size(0), -1, self.sizeScale0[0], self.sizeScale0[1])

    def forward(self, x, test_all_scales=False):
        output = []
        ## Normalize the input ?
        if self.normalizationLayer is not None:
            x = self.normalizationLayer(x)

        if self.dimOutput > 1:
            x = x.view(-1, num_flat_features(x))
        # format layer
        x = self.leakyRelu(self.formatLayer(x))
        x = self.tensor_view(x) # change for init size differnt than 8

        x = self.normalizationLayer(x)

        # Scale 0 (no upsampling)
        for convLayer in self.groupScale0:
            x = self.leakyRelu(convLayer(x))
            if self.normalizationLayer is not None:
                x = self.normalizationLayer(x)

        # Dirty, find a better way
        if self.alpha > 0 and len(self.scaleLayers) == 1:
            y = self.toRGBLayers[-2](x)
            y = self.upscale(y, self.scaleSizes[1])

        # Add scale 0 for multiple scale output
        if test_all_scales:
            output.append(self.toRGBLayers[0](x))
        
        # Upper scales
        scale = 0
        for scale, layerGroup in enumerate(self.scaleLayers, 0):
            # NEEDS REVIEW, output size probably incorrect
            x = self.upscale(x, self.scaleSizes[scale + 1])
            for convLayer in layerGroup:
                x = self.leakyRelu(convLayer(x))
                if self.normalizationLayer is not None:
                    x = self.normalizationLayer(x)

            # Add intermediate scales for multi-scale output
            if test_all_scales and scale <= len(self.scaleLayers) - 2:
                output.append(self.toRGBLayers[scale + 1](x))

            if self.alpha > 0 and scale == (len(self.scaleLayers) - 2):
                y = self.toRGBLayers[-2](x)
                y = self.upscale(y, self.scaleSizes[scale + 2])

        # To RGB (no alpha parameter for now)
        x = self.toRGBLayers[-1](x)

        # Blending with the lower resolution output when alpha > 0
        if self.alpha > 0:
            x = self.alpha * y + (1.0-self.alpha) * x

        if self.generationActivation is not None:
            x = self.generationActivation(x)

        # Add last scale to multi-scale output
        if test_all_scales and scale != 0:
            output.append(x)
        
        if test_all_scales:   
            return output
        else:
            return x



class DNet(nn.Module):

    def __init__(self,
                 depthScale0,
                 initBiasToZero=True,
                 leakyReluLeak=0.2,
                 sizeDecisionLayer=1,
                 miniBatchNormalization=False,
                 dimInput=1,
                 equalizedlR=True,
                 sizeScale0=16,
                 inputSizes=0,
                 nScales=1):
        r"""
        Build a discriminator for a progressive GAN model

        Args:

            - depthScale0 (int): depth of the lowest resolution scales
            - initBiasToZero (bool): should we set the bias to zero when a
                                    new scale is added
            - leakyReluLeak (float): leakyness of the leaky relu activation
                                    function
            - decisionActivation: activation function of the decision layer. If
                                  None it will be the identity function.
                                  For the training stage, it's advised to set
                                  this parameter to None and handle the
                                  activation function in the loss criterion.
            - sizeDecisionLayer: size of the decision layer. Will typically be
                                 greater than 2 when ACGAN is involved
            - miniBatchNormalization: do we apply the mini-batch normalization
                                      at the last scale ?
            - dimInput (int): 3 (RGB input), 1 (grey-scale input)
        """
        super(DNet, self).__init__()
        self.kernelSize = 3
        self.padding = 1
        # if type(sizeScale0) == tuple:
        #     self.sizeScale0 = sizeScale0[1]
        #     self.Scale0y = sizeScale0[0]

        # else:
        self.sizeScale0 = sizeScale0
        self.inputSizes = inputSizes
        # Initialization paramneters
        self.initBiasToZero = initBiasToZero
        self.equalizedlR = equalizedlR
        self.dimInput = dimInput

        # Initalize the scales
        self.scalesDepth = [depthScale0]
        self.depthScale0 = depthScale0
        self.scaleLayers = nn.ModuleList()

        self.mergeLayers = nn.ModuleList()
        # Initialize the last layer
        self.initDecisionLayer(sizeDecisionLayer)

        # MiniBatch norm
        self.miniBatchNormalization = miniBatchNormalization

        # Layer 0
        self.groupScaleZero = nn.ModuleList()
        self.fromRGBLayers = nn.ModuleList()
        self.initScale0Layer()


        # Initalize the upscaling parameters
        self.alpha = 0

        # Leaky relu activation
        self.leakyRelu = torch.nn.LeakyReLU(leakyReluLeak)

    def initScale0Layer(self):
        # Minibatch standard deviation
        dimEntryScale0 = self.depthScale0
        if self.miniBatchNormalization:
            dimEntryScale0 += 1

        self.fromRGBLayers.append(EqualizedConv2d(self.dimInput, self.depthScale0, 1,
                                                  equalized=self.equalizedlR,
                                                  initBiasToZero=self.initBiasToZero))
        self.groupScaleZero.append(EqualizedConv2d(dimEntryScale0, self.depthScale0,
                                                   self.kernelSize, padding=self.padding,
                                                   equalized=self.equalizedlR,
                                                   initBiasToZero=self.initBiasToZero))

        self.groupScaleZero.append(EqualizedLinear(self.sizeScale0[0] * self.sizeScale0[1] * self.depthScale0, # here we have to multiply times the initial size (8 for generating 4096 in 9 scales)
                                                   self.depthScale0,
                                                   equalized=self.equalizedlR,
                                                   initBiasToZero=self.initBiasToZero))

    def addScale(self, depthNewScale):
        if type(depthNewScale) is list:
            depthNewScale = depthNewScale[1]

        depthLastScale = self.scalesDepth[-1]
        self.scalesDepth.append(depthNewScale)
        self.scaleLayers.append(nn.ModuleList())

        self.scaleLayers[-1].append(EqualizedConv2d(depthNewScale,
                                                    depthNewScale,
                                                    self.kernelSize,
                                                    padding=self.padding,
                                                    equalized=self.equalizedlR,
                                                    initBiasToZero=self.initBiasToZero))
        self.scaleLayers[-1].append(EqualizedConv2d(depthNewScale,
                                                    depthLastScale,
                                                    self.kernelSize,
                                                    padding=self.padding,
                                                    equalized=self.equalizedlR,
                                                    initBiasToZero=self.initBiasToZero))

        self.fromRGBLayers.append(EqualizedConv2d(self.dimInput,
                                                  depthNewScale,
                                                  1,
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

        if not self.fromRGBLayers:
            raise AttributeError("Can't set an alpha layer if only the scale 0"
                                 "is defined")

        self.alpha = alpha

    def initDecisionLayer(self, sizeDecisionLayer):
        self.decisionLayer = EqualizedLinear(self.scalesDepth[0],
                                             sizeDecisionLayer,
                                             equalized=self.equalizedlR,
                                             initBiasToZero=self.initBiasToZero)


    def downScale(self, x, size=0):
        # return F.adaptive_avg_pool2d(x, output_size=size)
        return F.interpolate(x, mode='nearest', size=size)

    def forward(self, x, getFeature = False):

        # Alpha blending
        if self.alpha > 0 and len(self.fromRGBLayers) > 1:
            y = self.downScale(x, size=self.inputSizes[len(self.fromRGBLayers) - 2])
            y = self.leakyRelu(self.fromRGBLayers[-2](y))

        # From RGB layer
        x = self.leakyRelu(self.fromRGBLayers[-1](x))

        # Caution: we must explore the layers group in reverse order !
        # Explore all scales before 0
        mergeLayer = self.alpha > 0 and len(self.scaleLayers) > 1
        shift = len(self.fromRGBLayers) - 2

        for i, groupLayer in enumerate(reversed(self.scaleLayers)):
            for layer in groupLayer:
                x = self.leakyRelu(layer(x))
            x = self.downScale(x, size=self.inputSizes[shift])

            if mergeLayer:
                mergeLayer = False
                x = self.alpha * y + (1-self.alpha) * x

            shift -= 1
       
       # Now the scale 0
       # Minibatch standard deviation
        if self.miniBatchNormalization:
            x = miniBatchStdDev(x)
        
        x = self.leakyRelu(self.groupScaleZero[0](x))
        x = x.view(-1, num_flat_features(x))

        x_lin = self.groupScaleZero[1](x)
        x = self.leakyRelu(x_lin)

        out = self.decisionLayer(x)

        if not getFeature:
            return out

        return out, x_lin

