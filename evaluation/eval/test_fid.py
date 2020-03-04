import os
import json
import torchvision
import torch

from os.path import dirname, realpath, join
from ..metrics.inception_score import InceptionScore
from ..utils.utils import printProgressBar
from ..utils.utils import getVal, loadmodule, getLastCheckPoint, \
    parse_state_name, getNameAndPackage, saveScore
from ..networks.constant_net import FeatureTransform

from .train_inception_model import SpectrogramInception3
from ..datasets.datamanager import AudioDataManager

from ..metrics.frechet_inception_distance import FrechetInceptionDistance
import ipdb
from torch.utils.data import DataLoader
import numpy as np

def test(parser, visualisation=None):

    kwargs = vars(parser.parse_args())

    # Are all parameters available ?
    name = getVal(kwargs, "name", None)
    if name is None and not kwargs['selfNoise']:
        raise ValueError("You need to input a name")

    module = getVal(kwargs, "module", None)
    if module is None:
        raise ValueError("You need to input a module")

    # Loading the model
    scale = getVal(kwargs, "scale", None)

    if name is not None:
        iteration = getVal(kwargs, "iter", None)

        checkPointDir = os.path.join(kwargs["dir"], name)
        checkpointData = getLastCheckPoint(
            checkPointDir, name, scale=scale, iter=iteration)

        if checkpointData is None:
            print(scale, iteration)
            if scale is not None or iteration is not None:
                raise FileNotFoundError("Not checkpoint found for model "
                                        + name + " at directory " + dir +
                                        " for scale " + str(scale) +
                                        " at iteration " + str(iteration))
            raise FileNotFoundError(
                "Not checkpoint found for model " + name + " at directory "
                + dir)

        modelConfig, pathModel, _ = checkpointData
        with open(modelConfig, 'rb') as file:
            configData = json.load(file)

        modelPackage, modelName = getNameAndPackage(module)
        modelType = loadmodule(modelPackage, modelName)

        model = modelType(useGPU=True,
                          # storeAVG=True,
                          storeAVG=False,
                          **configData)

        if scale is None or iter is None:
            _, scale, iteration = parse_state_name(pathModel)

        print("Checkpoint found at scale %d, iter %d" % (scale, iteration))
        model.load(pathModel)

    elif scale is None:
        raise AttributeError("Please provide a scale to compute the noise of \
        the dataset")

    # Building the score instance
    # classifier = torchvision.models.inception_v3(pretrained=True).cuda()

    latentDim = model.config.categoryVectorDim
    # classifier = torchvision.models.inception_v3(pretrained=False,
    #                                              num_classes=128).cpu()
    state_dict = torch.load(join(dirname(realpath(__file__)), "inception_models/2019-08-01.pt"))
    classifier = SpectrogramInception3(109, aux_logits=False)
    classifier.load_state_dict(state_dict)

    scoreMaker = FrechetInceptionDistance()


    batchSize = 25
    nBatch = 1000
    path_to_raw = "/ldaphome/jnistal/data/nsynth-train/audio/"
    path_out = "/ldaphome/jnistal/sandbox"

    data_manager = AudioDataManager(path_to_raw=path_to_raw,
                                    path_out=path_out,
                                    db_name='nsynth',
                                    sample_rate=16000,
                                    audio_len=16000,
                                    data_type='AUDIO',
                                    transform='cqt',
                                    db_size=nBatch * batchSize,
                                    inscale_load=False,
                                    # labels=["mallet", "flute"],
                                    labels=["mallet", "flute", "keyboard", "guitar"],
                                    log=True,
                                    fold_cqt=True,
                                    n_bins=96,
                                    padding=True,
                                    load_metadata=True)
    data_loader = data_manager.get_loader()
    data_loader = DataLoader(data_loader,
                             batch_size=batchSize,
                             shuffle=True,
                             num_workers=2)

    print("Computing the inception score...")

    classifier = classifier.eval()

    real_acts = np.empty((nBatch * batchSize, 109))
    fake_acts = np.empty((nBatch * batchSize, 109))
    with torch.no_grad():
        for index in range(nBatch):
            start = index * batchSize
            end = start + batchSize

            data_iter = iter(data_loader)
            inputReal = data_iter.next()[0][:, 0:1]
            inputFake = model.test(model.buildNoiseData(batchSize)[0],
                                   toCPU=True, getAvG=True)[:, 0:1]
            # scoreMaker.updateWithMiniBatch(imgTransform(inputFake))
            
            real_acts[start:end] = classifier(inputReal).detach().cpu().numpy()
            fake_acts[start:end] = classifier(inputFake).detach().cpu().numpy()


    distance = scoreMaker.getDistance(real_acts, fake_acts)
    # distance = scoreMaker.getDistance(inputReal[0][:, 0:1], inputFake[:, 0:1])
    printProgressBar(index, nBatch)
    print(distance)

    printProgressBar(nBatch, nBatch)
    print("Merging the results, please wait it can take some time...")
    score = distance

    # Now printing the results
    print(score)

    # Saving the results
    if name is not None:
        outPath = os.path.join(checkPointDir, name + "_swd.json")
        saveScore(outPath, score,
                  scale, iteration)

