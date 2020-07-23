import math

import torch.nn as nn
import torch

from numpy import prod, sqrt


class NormalizationLayer(nn.Module):

    def __init__(self):
        super(NormalizationLayer, self).__init__()

    def forward(self, x, epsilon=1e-8):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


def Upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1:
        return x
    s = x.size()
    x = x.view(-1, s[1], s[2], 1, s[3], 1)
    x = x.expand(-1, s[1], s[2], factor, s[3], factor)
    x = x.contiguous().view(-1, s[1], s[2] * factor, s[3] * factor)
    return x


def getLayerNormalizationFactor(x):
    r"""
    Get He's constant for the given layer
    https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
    """
    size = x.weight.size()
    fan_in = prod(size[1:])

    return math.sqrt(2.0 / fan_in)



class ConstrainedLayer(nn.Module):
    r"""
    A handy refactor that allows the user to:
    - initialize one layer's bias to zero
    - apply He's initialization at runtime
    """

    def __init__(self,
                 module,
                 equalized=True,
                 lrMul=1.0,
                 initBiasToZero=True):
        r"""
        equalized (bool): if true, the layer's weight should evolve within
                         the range (-1, 1)
        initBiasToZero (bool): if true, bias will be initialized to zero
        """

        super(ConstrainedLayer, self).__init__()

        self.module = module
        self.equalized = equalized

        if initBiasToZero:
            self.module.bias.data.fill_(0)
        if self.equalized:
            self.module.weight.data.normal_(0, 1)
            self.module.weight.data /= lrMul
            self.weight = getLayerNormalizationFactor(self.module) * lrMul

    def forward(self, x, *args):

        x = self.module(x, *args)
        if self.equalized:
            x *= self.weight
        return x


class EqualizedConv2d(ConstrainedLayer):

    def __init__(self,
                 nChannelsPrevious,
                 nChannels,
                 kernelSize,
                 padding=0,
                 bias=True,
                 transposed=False,
                 groups=1,
                 **kwargs):
        r"""
        A nn.Conv2d module with specific constraints
        Args:
            nChannelsPrevious (int): number of channels in the previous layer
            nChannels (int): number of channels of the current layer
            kernelSize (int): size of the convolutional kernel
            padding (int): convolution's padding
            bias (bool): with bias ?
        """
        if transposed:
            ConstrainedLayer.__init__(self,
                                      nn.ConvTranspose2d(nChannelsPrevious, nChannels,
                                                kernelSize, padding=padding,
                                                bias=bias, groups=groups),
                                      **kwargs)
        else:
            ConstrainedLayer.__init__(self,
                                      nn.Conv2d(nChannelsPrevious, nChannels,
                                                kernelSize, padding=padding,
                                                bias=bias, groups=groups),
                                      **kwargs)


class AudioNorm(nn.Module):
    """

    """
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True)
                                  + 1e-8)


class Conv2DBlock(nn.Module):
    """


    """
    def __init__(
        self, in_channel, out_channel, kernel_size, padding, kernel_size2=None, 
        padding2=None, pixel_norm=True, spectral_norm=False
    ):
        super().__init__()

        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        self.conv = nn.Sequential(EqualizedConv2d(in_channel, out_channel,
                                            kernel1, padding=pad1),
                                nn.LeakyReLU(0.2),
                                EqualizedConv2d(out_channel, out_channel,
                                            kernel2, padding=pad2),
                                nn.LeakyReLU(0.2))

    def forward(self, input):

        out = self.conv(input)

        return out


class AdaptiveInstanceNorm2D(nn.Module):
    """

    """
    def __init__(self, in_channel, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_channel)

        # self.style = EqualLinear(style_dim, in_channel * 2)
        self.style = EqualizedLinear(style_dim, in_channel * 2)

        self.style.module.bias.data[:in_channel] = 1
        self.style.module.bias.data[in_channel:] = 0        

        # self.style.linear.bias.data[:in_channel] = 1
        # self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):

        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out


class EqualizedLinear(ConstrainedLayer):

    def __init__(self,
                 nChannelsPrevious,
                 nChannels,
                 bias=True,
                 **kwargs):
        r"""
        A nn.Linear module with specific constraints
        Args:
            nChannelsPrevious (int): number of channels in the previous layer
            nChannels (int): number of channels of the current layer
            bias (bool): with bias ?
        """

        ConstrainedLayer.__init__(self,
                                  nn.Linear(nChannelsPrevious, nChannels,
                                  bias=bias), **kwargs)




class ConstantInput2D(nn.Module):
    """

    """
    def __init__(self, channel, size):
        super().__init__()
        if type(size) in [tuple, list]:
            self.input = nn.Parameter(torch.randn(1, channel, size[0], size[1]))
        else:
            self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out

class NoiseInjection2D(nn.Module):
    """
    """
    def __init__(self, channel):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, image, noise):

        return image + self.weight * noise

class EqualizedNoiseInjection2D(ConstrainedLayer):
    """
    """
    def __init__(self, nChannels, **kargs):
        ConstrainedLayer.__init__(self, 
                                  NoiseInjection2D(nChannels),
                                  initBiasToZero=False,
                                  **kargs)

class DummyBlock(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(x, *args):
        return x


class GANsynthInitFormatLayer(nn.Module):
    def __init__(self, 
                 dimLatent, 
                 scaleDepth, 
                 outputShape, 
                 equalizedlR, 
                 initBiasToZero,
                 pixelNorm=True
                 ):
        super().__init__()
        self.module = nn.ModuleList()
        self.module.extend([
            torch.nn.ZeroPad2d((
                outputShape[1] - 1, 
                outputShape[1] - 1, 
                outputShape[0] - 1, 
                outputShape[0] - 1)),
            EqualizedConv2d(
                dimLatent,
                scaleDepth,
                outputShape,
                equalized=equalizedlR,
                initBiasToZero=initBiasToZero,
                padding=(0, 0)),
            nn.LeakyReLU(0.2),
            EqualizedConv2d(
                scaleDepth,
                scaleDepth,
                3,
                equalized=equalizedlR,
                initBiasToZero=initBiasToZero,
                padding=1),
            nn.LeakyReLU(0.2),
        ])
        if pixelNorm:
            self.module.insert(3, AudioNorm())
            self.module.insert(6, AudioNorm())

    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)
        for m in self.module:
            x = m(x)
        return x



