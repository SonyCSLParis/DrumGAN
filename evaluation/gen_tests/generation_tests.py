import numpy as np
import torch
from numpy.random import randint
from numpy import linspace


class StyleGEvaluationManager(object):
    def __init__(self, model, n_gen=20, get_avg=False):
        self.model = model
        self.att_manager = model.ClassificationCriterion
        self.n_gen = n_gen
        self.get_avg = get_avg
        # self.model.config.ac_gan = True

        self.ref_rand_z = self.model.buildNoiseData(self.n_gen, skipAtts=True)[0]

        # self.ref_rand_z = self.model.buildNoiseData(self.n_gen)[0]
        self.latent_noise_dim = self.model.config.noiseVectorDim
        self.att_dim = self.model.config.categoryVectorDim_G

        self.n_iterp_steps = 10

    def test_random_generation(self):
        gen_batch = self.model.test(self.ref_rand_z,
                               toCPU=True,
                               getAvG=self.get_avg)
        return gen_batch

    def test_single_pitch_random_z(self, pitch=55):
        input_z = self.ref_rand_z.clone()
        input_z[:, -self.att_dim:] = torch.zeros(self.att_dim)

        if "pitch" in self.att_manager.keyOrder:
            pitch_att_dict = self.att_manager.inputDict['pitch']
            pitch_att_indx = pitch_att_dict['order']
            pitch_att_size = self.att_manager.attribSize[pitch_att_indx]
            att_shift = 0
            for j, att in enumerate(self.att_manager.keyOrder):
                if att in self.att_manager.skipAttDfake: continue
                if att == "pitch": break
                att_shift += self.att_manager.attribSize[j]
            # att_shift = sum(self.att_manager.attribSize[:pitch_att_indx])
            try:
                pitch_indx = \
                    self.att_manager.inputDict['pitch']['values'].index(pitch)
            except ValueError:
                pitch_indx = randint(pitch_att_size)
            input_z[:, self.latent_noise_dim + att_shift + pitch_indx] = 1

        gen_batch = self.model.test(input_z, toCPU=True, getAvG=True)
        return gen_batch

    def test_single_z_pitch_sweep(self):
        if "pitch" not in self.att_manager.keyOrder: 
            raise AttributeError("Pitch not in the model's attributes")
        pitch_att_dict = self.att_manager.inputDict['pitch']
        pitch_att_indx = pitch_att_dict['order']
        pitch_att_size = self.att_manager.attribSize[pitch_att_indx]
        att_shift = 0
        for j, att in enumerate(self.att_manager.keyOrder):
            if att in self.att_manager.skipAttDfake: continue
            if att == "pitch": break
            att_shift += self.att_manager.attribSize[j]
        # att_shift = sum(self.att_manager.attribSize[:pitch_att_indx])
        input_z = []
        for i in range(pitch_att_size):
            z = self.ref_rand_z[0].clone()
            z[-self.att_dim:] = torch.zeros(self.att_dim)
            z[-self.att_dim + att_shift + i] = 1
            input_z.append(z)
        input_z = torch.stack(input_z)
        gen_batch = self.model.test(input_z, toCPU=True, getAvG=True)
        return gen_batch

    def test_single_pitch_latent_interpolation(self, pitch=55):
        z = self.ref_rand_z[:2, :].clone()
        if self.att_dim > 0:
            z[:, -self.att_dim:] = torch.zeros(self.att_dim)

        if "pitch" in self.att_manager.keyOrder:
            pitch_att_dict = self.att_manager.inputDict['pitch']
            pitch_att_indx = pitch_att_dict['order']
            pitch_att_size = self.att_manager.attribSize[pitch_att_indx]
            try:
                pitch_indx = \
                    self.att_manager.inputDict['pitch']['values'].index(pitch)
            except ValueError:
                pitch_indx = randint(pitch_att_size)
            
            z[:, self.latent_noise_dim + pitch_indx] = 1
        
        input_z = []
        for i in linspace(0., 1., self.n_iterp_steps, True):
            input_z.append((1-i)*z[0] + i*z[1])
            # z /= abs(z)
        input_z = torch.stack(input_z)
        gen_batch = self.model.test(input_z, toCPU=True, getAvG=True)
        return gen_batch

    def test_single_pitch_sph_latent_interpolation(self, pitch=55):
        def get_rand_gaussian_outlier(ndim):
            r = 3
            ph = 2 * np.pi * np.random.rand(ndim)
            cos = [np.cos(p) for p in ph]
            sin = [np.sin(p) for p in ph]
            vector = []
            for i in range(len(ph)):
                if i == 0:
                    vector += [cos[i]]
                elif i == len(ph):
                    vector += [np.prod(sin)]
                else:
                    vector += [np.prod(sin[:i]) * cos[i]]
            input_z = []
            for i in linspace(r, -1*r, self.n_iterp_steps, True):
                input_z.append(torch.from_numpy(np.multiply(i, vector).astype(float)))
            return input_z

        input_z = get_rand_gaussian_outlier(self.latent_noise_dim)
        if self.att_dim > 0:
            input_z = torch.stack(input_z).double()
            input_z = torch.cat([input_z, torch.zeros((input_z.size(0), self.att_dim)).double()], dim=1)

        if "pitch" in self.att_manager.keyOrder:
            pitch_att_dict = self.att_manager.inputDict['pitch']
            pitch_att_indx = pitch_att_dict['order']
            pitch_att_size = self.att_manager.attribSize[pitch_att_indx]
            try:
                pitch_indx = \
                    self.att_manager.inputDict['pitch']['values'].index(pitch)
            except ValueError:
                pitch_indx = randint(pitch_att_size)
            
            input_z[:, self.latent_noise_dim + pitch_indx] = 1
        # input_z = torch.stack(z)

        gen_batch = self.model.test(input_z.float(), toCPU=True, getAvG=True)
        return gen_batch

    def test_single_pitch_sph_surface_interpolation(self, pitch=55):

        r = 1
        ph = 2 * np.pi * np.random.rand(self.latent_noise_dim)
        input_z = []
        for i in range(30):
            phi = ph + 2*np.pi*i/30

            cos = [np.cos(p) for p in ph]
            sin = [np.sin(p) for p in ph]
            vector = []
            for i in range(len(ph)):
                if i == 0:
                    vector += [cos[i]]
                elif i == len(ph):
                    vector += [np.prod(sin)]
                else:
                    vector += [np.prod(sin[:i]) * cos[i]]
            zi = np.multiply(r, vector)
            input_z.append(torch.from_numpy(zi.astype(float)))

        if self.att_dim > 0:

            input_z = torch.stack(input_z).double()
            input_z = torch.cat([input_z, torch.zeros((input_z.size(0), self.att_dim)).double()], dim=1)

        if "pitch" in self.att_manager.keyOrder:
            pitch_att_dict = self.att_manager.inputDict['pitch']
            pitch_att_indx = pitch_att_dict['order']
            pitch_att_size = self.att_manager.attribSize[pitch_att_indx]
            try:
                pitch_indx = \
                    self.att_manager.inputDict['pitch']['values'].index(pitch)
            except ValueError:
                pitch_indx = randint(pitch_att_size)
            
            input_z[:, self.latent_noise_dim + pitch_indx] = 1
        # input_z = torch.stack(input_z)

        gen_batch = self.model.test(input_z.float(), toCPU=True, getAvG=True)
        return gen_batch
