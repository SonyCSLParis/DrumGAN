import os
import ipdb
import json

import argparse


from os.path import dirname, realpath, join
# from ..metrics.inception_score import InceptionScore
# from ..utils.utils import printProgressBar
# from ..utils.utils import getVal, loadmodule, getLastCheckPoint, \
#     parse_state_name, getNameAndPackage, saveScore
# from ..networks.constant_net import FeatureTransform

# from .train_inception_model import SpectrogramInception3
# from ..datasets.datamanager import AudioDataManager

# from ..metrics.frechet_inception_distance import FrechetInceptionDistance
import sys
sys.path.append("/Users/javier/Developer/gans/gan_zoo_audio/models/metrics")
from kernel_inception_distance import polynomial_mmd

import numpy as np
import matplotlib.pyplot as plt 
from tools import mkdir_in_path, list_files_abs_path
from essentia import array
import essentia.standard as es
import matplotlib.pyplot as plt



SAMPLE_RATE = 44100

def compute_kernel(x, y, k):
    batch_size_x, dim_x = x.size()
    batch_size_y, dim_y = y.size()
    assert dim_x == dim_y

    xx = x.unsqueeze(1).expand(batch_size_x, batch_size_y, dim_x)
    yy = y.unsqueeze(0).expand(batch_size_x, batch_size_y, dim_y)
    distances = (xx - yy).pow(2).sum(2)
    return k(distances)

def mmd(z_tilde, z):
    # gaussian
    def gaussian(d, var=16.):
        return torch.exp(- d / var).sum(1).sum(0)

    # inverse multiquadratics
    def inverse_multiquadratics(d, var=16.):
        """
        :param d: (num_samples x, num_samples y)
        :param var:
        :return:
        """
        return (var / (var + d)).sum(1).sum(0)

    k = inverse_multiquadratics
    # k = gaussian
    batch_size = z_tilde.size(0)
    zz_ker = compute_kernel(z, z, k)
    z_tilde_z_tilde_ker = compute_kernel(z_tilde, z_tilde, k)
    z_z_tilde_ker = compute_kernel(z, z_tilde, k)

    first_coefs = 1. / (batch_size * (batch_size - 1))
    second_coef = 2 / (batch_size * batch_size)
    mmd = (first_coefs * zz_ker
           + first_coefs * z_tilde_z_tilde_ker
           - second_coef * z_z_tilde_ker)
    return mmd

def load_audios(path):
    return map(lambda x: es.MonoLoader(filename=x, sampleRate=SAMPLE_RATE), list_files_abs_path(path))

def extract_essentia_feats(audios):
    # extractor = es.Extractor(rhythm=False, sampleRate=SAMPLE_RATE, relativeIoi=False, namespace="FeatureTransform", tonalFrameSize=2048, tonalHopSize=128, dynamics=False)
    # extractor = es.LowLevelSpectralExtractor(sampleRate=16000)
    extractor = es.MusicExtractor(analysisSampleRate=SAMPLE_RATE, endTime=1)
    # return map(lambda x: extractor(x()), audios)
    return map(lambda x: extractor(x), audios)

def get_feature_dict(feat_pools, index=1):
    feat_dict = {}
    feat_lists = []

    for pools in feat_pools:
        feat = []
        for feature in pools[index].descriptorNames():
            # skip metadata and rhythmic features
            if "rhythm" in feature or "metadata" in feature: continue
            # if feature not in dict, initialize
            if feature not in feat_dict.keys():
                # if feature is an array (eg MFCCs) convert to different subfeatures
                if type(pools[index][feature]) is np.ndarray:
                    feat_dict[feature] = {}
                    if type(pools[index][feature][0]) is not np.ndarray:
                        for i in range(len(pools[index][feature])):
                            feat_dict[feature][str(i)] = []
                    else:
                        # This means it's a list of spectrogrma-based feature lists
                        # Consider extracting mean and variances?
                        continue
                else:
                    feat_dict[feature] = []

            if type(pools[index][feature]) is np.ndarray:
                if type(pools[index][feature][0]) is not np.ndarray:
                    for i in range(len(pools[index][feature])):
                        feat_dict[feature][str(i)].append(pools[index][feature][i])
                        
                        # Skip string features
                        if type(pools[index][feature][i]) is not str:
                            feat.append(pools[index][feature][i])
                else: continue
            else:
                feat_dict[feature].append(pools[index][feature])
                # Skip string features
                if type(pools[index][feature]) is not str:
                    feat.append(pools[index][feature])
        feat_lists.append(feat)
    return feat_dict, feat_lists


def plot_histogram(histogram, x_label="bins", title="default"):
    fig, ax = plt.subplots()
    ax.bar(range(len(histogram)), histogram, width=1)
    ax.bar(range(len(histogram)), histogram, width=1)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Frequency')
    plt.title(title)
    ax.set_xticks([20 * x + 0.5 for x in range(int(len(histogram) / 20))])
    ax.set_xticklabels([str(20 * x) for x in range(int(len(histogram) / 20))])
    plt.show()

def compute_histogram(feat_dict):
    histogram = es.Histogram(normalize='unit_sum', numberBins=20)

    for key, val in feat_dict.items():
        try:
            if type(val) is dict:
                for skey, sval in val.items():
                    hist, bins = histogram(array(sval))
                    plot_histogram(hist, x_label="{0}_{1}".format(key, skey), title='{0}.{1} histogram'.format(key, skey))
            else:
                hist, bins = histogram(array(val))
                plot_histogram(hist, key, '{0} histogram'.format(key))
        except Exception as e:
            print(e)
            print("Exception while processing feature: {}".format(key))
            continue

def compare_histograms(rfeat_dict, ffeat_dict):
    for key, val in rfeat_dict.items():
        try:
            if type(val) is dict:
                for skey, sval in val.items():
                    # hist, bins = histogram(array(sval))
                    fake_val = ffeat_dict[key][skey]
                    # bins = np.linspace(-10, 10, 100)
                    _min = min(min(fake_val), min(sval), max(fake_val), max(sval),)
                    _max = max(max(fake_val), max(sval), min(fake_val), min(sval))
                    plt.hist([(np.array(sval) - _min)/(_max - _min), (np.array(fake_val) - _min)/(_max - _min)], 
                             range=(0, 1),
                             alpha=0.5,
                             label=['real_data', 'fake_data'],
                             # bins=10,
                             density=True,
                             stacked=False)
                    # plt.hist((sval - _min)/(_max - _min), 10, range=(_min, _max), alpha=0.5, label='real_data', density=True)
                    # plt.hist((fake_val - _min)/(_max - _min), 10, range=(_min, _max), alpha=0.5, label='fake_data', density=True)
                    plt.legend(loc='upper right')
                    plt.title("{0}.{1}".format(key, skey))
                    plt.show()
            else:
                fake_val = ffeat_dict[key]
                # bins = np.linspace(-10, 10, 100)
                _min = min(min(fake_val), min(val), max(fake_val), max(val))
                _max = max(max(fake_val), max(val), min(fake_val), min(val))
                plt.hist([(np.array(val) - _min)/(_max - _min), (np.array(fake_val) - _min)/(_max - _min)], 
                             range=(0, 1),
                             alpha=0.5,
                             label=['real_data', 'fake_data'],
                             # bins=10,
                             density=True,
                             stacked=False)

                # plt.hist((np.array(val)-_min)/(_max-_min), 10, range=(_min, _max), alpha=0.5, label='real_data', density=True)
                # plt.hist((np.array(fake_val) - _min)/(_max-_min), 10, range=(_min, _max), alpha=0.5, label='fake_data', density=True)
                plt.legend(loc='upper right')
                plt.title(key)
                plt.show()
        except Exception as e:
            ipdb.set_trace()
            print(e)
            print("Exception while processing feature: {}".format(key))
            continue


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Testing script')
    parser.add_argument('-r', '--real-path', type=str, dest='real_path',
                         help='Path to folder containing real examples')
    
    parser.add_argument('-f', '--fake-path', dest='fake_path', help='Path to fake examples',
                        type=str)
    parser.add_argument('--np_vis', help=' Replace visdom by a numpy based \
                        visualizer (SLURM)',
                        action='store_true')

    args = parser.parse_args()
    real_audios = load_audios(args.real_path)
    fake_audios = load_audios(args.fake_path)
    
    real_features = extract_essentia_feats(list_files_abs_path(args.real_path))
    fake_features = extract_essentia_feats(list_files_abs_path(args.fake_path))
    
    real_featDict, real_featList = get_feature_dict(real_features, index=0)
    fake_featDict, fake_featList = get_feature_dict(fake_features, index=0)

    ipdb.set_trace()
    n_examples = min(len(real_featList), len(fake_featList))
    print("Extracting polynomial_mmd distance:")
    mmd_loss = mmd(real_featList[:n_examples], fake_featList[:n_examples])
    
    print(mmd_loss)

    compare_histograms(real_featDict, fake_featDict)
    # compute_histogram(real_featDict)
    # compute_histogram(fake_featDict)

