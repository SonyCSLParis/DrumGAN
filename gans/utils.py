
import librosa
import librosa.display as display
import matplotlib.pyplot as plt
import os
import numpy as np

import torch.nn.functional as F


def librosaSpec(x):
    return librosa.magphase(x)

def save_spectrogram(out_dir=".", fn="spect.png", spect_compl=None):
    plt.clf()

    spect = spect_compl[0] + 1j * spect_compl[1]
    mag, ph = librosa.magphase(spect)

    display.specshow(np.log(mag), y_axis='linear', sr=16000, hop_length=256,
                     fmax=7900, fmin=10)
    plt.colorbar(format='%+2.0f dB')
    plt.savefig(os.path.join(out_dir, "mag_" + fn))

    plt.clf()
    display.specshow(np.angle(ph), y_axis='linear', sr=16000, hop_length=256,
                     fmax=7900, fmin=10)
    plt.colorbar(format='%+2.0f dB')
    plt.savefig(os.path.join(out_dir, "ph_" + fn))

def scale_interp(x, size=0, mode='nearest'):
    # return F.adaptive_avg_pool2d(x, output_size=size)
    return F.interpolate(x, mode=mode, size=size)


# def interpolate(x, y, batch=True):
#     assert x.size() == y.size(), "real and fake must be the same size"
#     assert x.device == y.device, "tensor are not on the same device"
#     if not batch:
#         alpha = torch.rand([]).to(x.device)
#         return alpha * x + (1-alpha) * y
#     else:


#         alpha = getattr(torch.rand(x.size(0), 1, 1), str(x.device))()
#         # alpha = alpha.expand(*x.size()[1:], x.size(0)).permute(x.dim()-1, *list(range(x.dim()-1)))
#         # alpha = alpha.to(x.device)
#         return alpha * x.data + (1-alpha.data) * y



# def gradient_norm(x_hat_predict, x_t):
#     one = torch.ones(x_hat_predict.size(), device=x_hat_predict.device)
#     gradient = torch.autograd.grad(outputs=x_hat_predict.sum(), inputs=x_t, create_graph=True)[0]

#     # gradient = torch.autograd.grad(outputs=x_hat_predict, inputs=x_t,grad_outputs=one,
#     #                                create_graph=False, retain_graph=False,
#     #                                only_inputs=True)[0]
#     grad_norm = gradient.view(x_t.size(0), -1).norm(p=2, dim=1)
#     return grad_norm


# def gradient_penalty(grad_n):
#     grad_pen = ((grad_n - 1)**2).mean()
#     grad_pen = 10 * grad_pen
#     return grad_pen



# class GeneratorLoss(nn.Module):
#     def __init__(self, gen, dis, representation=None):
#         super(GeneratorLoss, self).__init__()
#         self.gen = gen
#         self.dis = dis
#         self.representation = representation

#     def forward(self, input, noise: torch.Tensor, step, alpha):
#         fake = self.gen(input=input, noise=noise, step=step, alpha=alpha)
#         # fake = self.representation(fake)
#         fake_loss = self.dis(fake).mean()

#         return fake_loss


# class DiscriminatorLoss(nn.Module):
#     def __init__(self, gen, dis, representation=None):
#         super(DiscriminatorLoss, self).__init__()
#         self.gent = gen
#         self.dis = dis
#         self.representation = representation
#         self.step = 0
#         self.alpha = 0

#     def lipschitz_norm(self, real, fake, penalty=True):
#         # n = min(100, real.size(0))
#         # x_t = interpolate(real[:n], fake[:n])
#         x_t = interpolate(real, fake)
#         x_t.requires_grad = True
#         x_t_predict = self.dis(x_t, step=self.step, alpha=self.alpha)

#         grad_norm = gradient_norm(x_t_predict, x_t)
#         if penalty:
#             grad_norm = gradient_penalty(grad_norm)
#         grad_max = grad_norm.max()
#         return grad_max

#     def forward(self, x: torch.Tensor, noise: torch.Tensor, step, alpha, lipschitz_norm=True):
#         # real = self.representation(x)
#         real = x
#         self.step = step
#         self.alpha = alpha

#         real_loss = self.dis(real, step=step, alpha=alpha).mean()

#         # with torch.no_grad():
#         #     fake = self.gent(input, noise, step=step, alpha=alpha)
#         fake = self.gent(noise, None, step=step, alpha=alpha)
#             # fake = self.representation(fake)
#         fake_loss = self.dis(fake, step=step, alpha=alpha).mean()

#         if lipschitz_norm:
#             lip_norm = self.lipschitz_norm(real, fake)
#         else:
#             lip_norm = None
#         return real_loss, fake_loss, lip_norm
