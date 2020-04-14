

import matplotlib.pyplot as plt

from .progressive_gan_trainer import ProgressiveGANTrainer

from .transform_style_gan import TStyleGAN
from tqdm import tqdm
from .spgan_config import _C
import numpy as np
import torch
from utils.utils import mkdir_in_path, saveAudioBatch


class TStyleGANTrainer(ProgressiveGANTrainer):
    r"""
    A class managing a progressive GAN training. Logs, chekpoints, visualization,
    and number iterations are managed here.
    """
    _defaultConfig = _C

    def getDefaultConfig(self):
        return TStyleGANTrainer._defaultConfig

    def __init__(self, **kwargs):
        r"""
        Args:
            - pathdb (string): path to the directorty containing the image
                               dataset
            - useGPU (bool): set to True if you want to use the available GPUs
                             for the training procedure
            - visualisation (module): if not None, a visualisation module to
                                      follow the evolution of the training
            - lossIterEvaluation (int): size of the interval on which the
                                        model'sloss will be evaluated
            - saveIter (int): frequency at which at checkpoint should be saved
                              (relevant only if modelLabel != None)
            - checkPointDir (string): if not None, directory where the checkpoints
                                      should be saved
            - modelLabel (string): name of the model
            - config (dictionary): configuration dictionnary. See std_p_gan_config.py
                                   for all the possible options
            - numWorkers (int): number of GOU to use. Will be set to one if not
                                useGPU
            - stopOnShitStorm (bool): should we stop the training if a diverging
                                     behavior is detected ?
        """

        ProgressiveGANTrainer.__init__(self, **kwargs)

    def initModel(self):
        r"""
        Initialize the GAN model.
        """
        print("Init StyleGAN")
        config = self.initScaleShapes()
        self.model = TStyleGAN(useGPU=self.useGPU, **config)

    def init_reference_eval_vectors(self, batch_size=50):

        self.true_ref, self.true_pair = self.loader.get_validation_set(batch_size)
        if self.useGPU:
            self.true_ref = self.true_ref.cuda()
            self.true_pair = self.true_pair.cuda()

        batch_size = min(batch_size, len(self.true_pair))
        self.ref_z, _ = self.model.buildNoiseData(batch_size)

    def trainOnEpoch(self,
                     dbLoader,
                     scale,
                     maxIter=-1):
        r"""
        Train the model on one epoch.

        Args:

            - dbLoader (DataLoader): dataset on which the training will be made
            - scale (int): scale at which is the training is performed
            - shiftIter (int): shift to apply to the iteration index when
                               looking for the next update of the alpha
                               coefficient
            - maxIter (int): if > 0, iteration at which the training should stop

        Returns:

            True if the training went smoothly
            False if a diverging behavior was detected and the training had to
            be stopped
        """
        with tqdm(dbLoader, desc='Iter-loop') as t:

            for x, y in t:
                # run evaluation/tests
                if self.iter % self.eval_i == 0: # and self.iter != 0:
                    self.run_tests_evaluation_and_visualization(scale)

                # Additionnal updates inside a scale
                # x = self.inScaleUpdate(self.iter, scale, x)
                # Optimize parameters
                allLosses = self.model.optimizeParameters(x, y, iter=self.iter)
                # Update and print losses
                self.updateRunningLosses(allLosses)
                state_msg = f'Iter: {self.iter} '
                for key, val in allLosses.items():
                    state_msg += f'{key}: {val:.2f}; '
                t.set_description(state_msg)

                # Plot losses
                if self.iter % self.loss_plot_i == 0:
                    # Reinitialize the losses
                    self.updateLossProfile(self.iter)
                    self.resetRunningLosses()
                    self.publish_loss()

                # Save checkpoint
                if self.iter % (self.saveIter - 1) == 0 and self.iter != 0:
                    labelSave = self.modelLabel + ("_s%d_i%d" % (scale, self.iter))
                    # Save Checkpoint
                    self.saveCheckpoint(outDir=self.checkPointDir,
                                        outLabel=labelSave,
                                        scale=scale,
                                        iter=self.iter)
                self.iter += 1
                if self.iter == maxIter:
                    return True
        self.epoch += 1
        return True

    def test_GAN(self):
        # sample fake data

        fake = self.model.test_G(z=self.ref_z, x=self.true_pair.float(),
                                 getAvG=False, toCPU=not self.useGPU)
        fake_avg = self.model.test_G(z=self.ref_z, x=self.true_pair.float(),
                                     getAvG=True, toCPU=not self.useGPU)
        
        # predict labels for fake data
        input_D = torch.cat([self.true_ref, fake, fake - self.true_ref], dim=1)
        D_fake, fake_emb = self.model.test_D(
            input_D, output_device='cpu', get_labels=False)

        input_D2 = torch.cat([self.true_ref, fake_avg, fake_avg - self.true_ref], dim=1)
        D_fake_avg, fake_avg_emb = self.model.test_D(
            input_D2, output_device='cpu', get_labels=False)
        
        # predict labels for true data
        # true, _ = self.loader.get_validation_set(len(self.ref_z), process=True)
        input_true = torch.cat([self.true_ref, self.true_pair.float(), self.true_pair.float() - self.true_ref], dim=1)
        D_true, true_emb = self.model.test_D(
            input_true, output_device='cpu', get_labels=False)

        return D_true, true_emb.detach(), \
               D_fake, fake_emb.detach(), \
               D_fake_avg, fake_avg_emb.detach(), \
               self.true_ref, fake.detach(), fake_avg.detach()

    def getDefaultIF(self, batch_size, win_size, hop_size):
        defaults = [(f+1) * hop_size / win_size % 1 for f in range(win_size // 2)]
        defaults = torch.Tensor(defaults)
        defaults[defaults > .5] = defaults[defaults > .5] - 1
        defaults *= 2 * np.pi
        if torch.cuda.is_available():
            defaults = defaults.cuda()
        defaults = defaults[None, None, :, None]
        return defaults.repeat((batch_size, 1, 1, 1))


    def run_default_if_test(self):
        scale_output_dir = mkdir_in_path(self.output_dir, f'scale_20')
        iter_output_dir = mkdir_in_path(scale_output_dir, f'iter_{self.iter}')
        output_dir = mkdir_in_path(iter_output_dir, 'generation')
        win_size = 16384
        hop_size = 2048
        batch_size = self.true_pair.shape[0]
        default_map_IF = self.getDefaultIF(batch_size, win_size,
                                           hop_size)
        default_map_IF[:, :, :, 0] = torch.Tensor(np.random.random(
            (batch_size, 1, default_map_IF.shape[3])) * 2 * np.pi - np.pi)
        # default_map_IF += torch.Tensor(
        #     np.random.normal(0, 0.1, default_map_IF.shape)).cuda()
        self.true_pair[:, 1:2, :, :] = default_map_IF
        batch_signal = self.loader.postprocess(self.true_pair)

        plt.clf()
        plt.plot(batch_signal[0][:])
        plt.show()
        plt.clf()
        plt.plot(batch_signal[0][:2048])
        plt.show()

        saveAudioBatch(
            self.loader.postprocess(self.true_ref),
            path=output_dir,
            basename=f'true_audio',
            overwrite=True)

        saveAudioBatch(
            self.loader.postprocess(self.true_pair),
            path=output_dir,
            basename=f'processed_audio',
            overwrite=True)


    def run_tests_evaluation_and_visualization(self, scale):
        scale_output_dir = mkdir_in_path(self.output_dir, f'scale_{scale}')
        iter_output_dir  = mkdir_in_path(scale_output_dir, f'iter_{self.iter}')

        with torch.no_grad():
            _, true_emb, \
            _, fake_emb, \
            _, fake_avg_emb, \
            true, fake, fake_avg = self.test_GAN()

            if self.save_gen:
                output_dir = mkdir_in_path(iter_output_dir, 'generation')
                if False:
                    win_size = 1024
                    hop_size = 256
                    batch_size = fake.shape[0]
                    default_map_IF = self.getDefaultIF(batch_size, win_size, hop_size)
                    default_map_IF[:, :, :, 0] = torch.Tensor(np.random.random((batch_size, 1, default_map_IF.shape[3])) * 2*np.pi - np.pi)
                    default_map_IF += torch.Tensor(np.random.normal(0, 0.1, default_map_IF.shape)).cuda()
                    fake[:, 0, :, :] = -np.inf
                    fake[0, 0, 0, :] = 5
                    fake[0, 0, 10, :] = 5
                    fake[1, 0, 17, :] = 5
                    fake[1, 0, 251, :] = 5
                    fake[1, 0, 7, :] = 5
                    fake[:, 1:2, :, :] = default_map_IF
                    #true[:, 1:2, :, :] = default_map_IF
                    # self.true_pair[:, 0, 3:, :] = -np.inf
                    # self.true_pair[:, 0, :2, :] = -np.inf
                    self.true_pair[:, 1:2, :, :] = default_map_IF
                    print(fake[0, 0, 100, :10])
                    print(fake[0, 1, 100, :10])
                    batch_signal = self.loader.postprocess(self.true_pair)

                    plt.clf()
                    plt.plot(batch_signal[0][:])
                    plt.show()
                    plt.clf()
                    plt.plot(batch_signal[0][:2048])
                    plt.show()
                saveAudioBatch(
                    self.loader.postprocess(fake),
                    path=output_dir,
                    basename=f'gen_audio_scale_{scale}',
                    overwrite=True)

                saveAudioBatch(
                    self.loader.postprocess(true),
                    path=output_dir,
                    basename=f'true_audio_scale_{scale}',
                    overwrite=True)

                saveAudioBatch(
                    self.loader.postprocess(self.true_pair),
                    path=output_dir,
                    basename=f'pair_audio_scale_{scale}',
                    overwrite=True)

            # if self.vis_manager != None:
            #     output_dir = mkdir_in_path(iter_output_dir, 'audio_plots')
            #     self.vis_manager.set_postprocessing(
            #         self.loader.get_postprocessor())
            #     self.vis_manager.publish(
            #         true[:5],
            #         labels=[],
            #         name=f'real_scale_{scale}',
            #         output_dir=output_dir)
            #     self.vis_manager.publish(
            #         fake[:5],
            #         labels=[],
            #         name=f'gen_scale_{scale}',
            #         output_dir=output_dir)
