import os
import numpy as np
import time
import json
from math import inf
from librosa.output import write_wav
import torch
from torch.nn.functional import interpolate

def checkexists_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        return False
    else:
        return True

def mkdir_in_path(path, dirname):
    dirpath = os.path.join(path, dirname)
    checkexists_mkdir(dirpath)
    return dirpath

def read_json(path):
    assert path.endswith('.json'), \
       f"Path {path} is not a JSON file."
    assert os.path.exists(path), \
       f"Path {path} doesn't exist!"
    with open(path) as file:
        data = json.load(file)
    file.close()
    return data

def filter_files_in_path(dir_path, format='.wav'):
    return filter(lambda x: x.endswith(format), os.listdir(dir_path))

def list_files_abs_path(dir_path, format='.wav'):
    return [os.path.join(os.path.abspath(dir_path), x) for x in filter_files_in_path(dir_path, format)]

def filter_keys_in_strings(strings, keys):
    return filter(lambda x: any([k in x for k in keys]), strings)

def get_filename(abs_path):
    return os.path.splitext(os.path.basename(abs_path))[0]

def isinf(tensor):
    r"""Returns a new tensor with boolean elements representing if each element
    is `+/-INF` or not.

    Arguments:
        tensor (Tensor): A tensor to check

    Returns:
        Tensor: A ``torch.ByteTensor`` containing a 1 at each location of
        `+/-INF` elements and 0 otherwise

    Example::

        >>> torch.isinf(torch.Tensor([1, float('inf'), 2,
                            float('-inf'), float('nan')]))
        tensor([ 0,  1,  0,  1,  0], dtype=torch.uint8)
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("The argument is not a tensor", str(tensor))
    return tensor.abs() == inf


def isnan(tensor):
    r"""Returns a new tensor with boolean elements representing if each element
    is `NaN` or not.

    Arguments:
        tensor (Tensor): A tensor to check

    Returns:
        Tensor: A ``torch.ByteTensor`` containing a 1 at each location of `NaN`
        elements.

    Example::

        >>> torch.isnan(torch.tensor([1, float('nan'), 2]))
        tensor([ 0,  1,  0], dtype=torch.uint8)
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("The argument is not a tensor", str(tensor))
    return tensor != tensor


def finiteCheck(parameters):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    for p in parameters:
        infGrads = isinf(p.grad.data)
        p.grad.data[infGrads] = 0

        nanGrads = isnan(p.grad.data)
        p.grad.data[nanGrads] = 0


def prepareClassifier(module, outFeatures):

    model = module()
    inFeatures = model.fc.in_features
    model.fc = torch.nn.Linear(inFeatures, outFeatures)

    return model


def getMinOccurence(inputDict, value, default):

    keys = list(inputDict.keys())
    outKeys = [x for x in keys if x <= value]
    outKeys.sort()

    if len(outKeys) == 0:
        return default

    return inputDict[outKeys[-1]]


def getNameAndPackage(strCode):

    if strCode == 'PGAN':
        return "progressive_gan", "ProgressiveGAN"

    if strCode == 'PPGAN':
        return "pp_gan", "PPGAN"

    if strCode == "DCGAN":
        return "DCGAN", "DCGAN"

    if strCode == "StyleGAN":
        return "style_progressive_gan", "StyleProgressiveGAN"

    raise ValueError("Unrecognized code " + strCode)


def parse_state_name(path):
    r"""
    Parse a file name with the given pattern:
    pattern = ($model_name)_s($scale)_i($iteration).pt

    Returns: None if the path doesn't fulfill the pattern
    """
    path = os.path.splitext(os.path.basename(path))[0]

    data = path.split('_')

    if len(data) < 3:
        return None

    # Iteration
    if data[-1][0] == "i" and data[-1][1:].isdigit():
        iteration = int(data[-1][1:])
    else:
        return None

    if data[-2][0] == "s" and data[-2][1:].isdigit():
        scale = int(data[-2][1:])
    else:
        return None

    name = "_".join(data[:-2])

    return name, scale, iteration


def parse_config_name(path):
    r"""
    Parse a file name with the given pattern:
    pattern = ($model_name)_train_config.json

    Raise an error if the pattern doesn't match
    """

    path = os.path.basename(path)

    if len(path) < 18 or path[-18:] != "_train_config.json":
        raise ValueError("Invalid configuration path")

    return path[:-18]


def getLastCheckPoint(dir, name, scale=None, iter=None):
    r"""
    Get the last checkpoint of the model with name @param name detected in the
    directory (@param dir)

    Returns:
    trainConfig, pathModel, pathTmpData

    trainConfig: path to the training configuration (.json)
    pathModel: path to the model's weight data (.pt)
    pathTmpData: path to the temporary configuration (.json)
    """
    trainConfig = os.path.join(dir, name + "_train_config.json")

    if not os.path.isfile(trainConfig):
        print("Checkpoint not found!")
        return None

    listFiles = [f for f in os.listdir(dir) if (
        os.path.splitext(f)[1] == ".pt" and
        parse_state_name(f) is not None and
        parse_state_name(f)[0] == name)]

    if scale is not None:
        listFiles = [f for f in listFiles if parse_state_name(f)[1] == scale]

    if iter is not None:
        listFiles = [f for f in listFiles if parse_state_name(f)[2] == iter]

    listFiles.sort(reverse=True, key=lambda x: (
        parse_state_name(x)[1], parse_state_name(x)[2]))

    if len(listFiles) == 0:
        print("Checkpoint not found!")
        return None

    pathModel = os.path.join(dir, listFiles[0])
    pathTmpData = os.path.splitext(pathModel)[0] + "_tmp_config.json"

    if not os.path.isfile(pathTmpData):
        print("Checkpoint not found!")
        return None

    return trainConfig, pathModel, pathTmpData


def getVal(kwargs, key, default):

    out = kwargs.get(key, default)
    if out is None:
        return default

    return out


def toStrKey(item):

    if item is None:
        return ""

    out = "_" + str(item)
    out = out.replace("'", "")
    return out


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def printProgressBar(iteration,
                     total,
                     prefix='',
                     suffix='',
                     decimals=1,
                     length=100,
                     fill='#'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent
                                  complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def loadPartOfStateDict(module, state_dict, forbiddenLayers=None):
    r"""
    Load the input state dict to the module except for the weights corresponding
    to one of the forbidden layers
    """
    own_state = module.state_dict()
    if forbiddenLayers is None:
        forbiddenLayers = []
    for name, param in state_dict.items():
        if name.split(".")[0] in forbiddenLayers:
            continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data

        own_state[name].copy_(param)


def loadStateDictCompatible(module, state_dict):
    r"""
    Load the input state dict to the module except for the weights corresponding
    to one of the forbidden layers
    """
    own_state = module.state_dict()
    for name, param in state_dict.items():
        if 'module' in name[:7]:
            name = name[7:]
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data

        if name in own_state:
            own_state[name].copy_(param)
            continue

        # Else see if the input name is a prefix
        suffixes = ["bias", "weight"]
        found = False
        for suffix in suffixes:
            indexEnd = name.find(suffix)
            if indexEnd > 0:
                newKey = name[:indexEnd] + "module." + suffix
                if newKey in own_state:
                    own_state[newKey].copy_(param)
                    found = True
                    break

        if not found:
            raise AttributeError("Unknow key " + name)


def loadmodule(package, name, prefix='..'):
    r"""
    A dirty hack to load a module from a string input

    Args:
        package (string): package name
        name (string): module name

    Returns:
        A pointer to the loaded module
    """
    strCmd = "from " + prefix + package + " import " + name + " as module"
    exec(strCmd)
    return eval('module')


def saveScore(outPath, outValue, *args):

    flagPath = outPath + ".flag"

    while os.path.isfile(flagPath):
        time.sleep(1)

    open(flagPath, 'a').close()

    if os.path.isfile(outPath):
        with open(outPath, 'rb') as file:
            outDict = json.load(file)
        if not isinstance(outDict, dict):
            outDict = {}
    else:
        outDict = {}

    fullDict = outDict

    for item in args[:-1]:
        if str(item) not in outDict:
            outDict[str(item)] = {}
        outDict = outDict[str(item)]

    outDict[args[-1]] = outValue

    with open(outPath, 'w') as file:
        json.dump(fullDict, file, indent=2)

    os.remove(flagPath)

def GPU_is_available():
    cuda_available = torch.cuda.is_available()
    if not cuda_available: print("Cuda not available. Running on CPU")
    return cuda_available


def load_model_checkp(dir, iteration=None, scale=None, **kwargs):
    # Loading the modelPackage
    from pg_gan.progressive_gan import ProgressiveGAN
    name = os.path.basename(dir)
    config_path = os.path.join(dir, f'{name}_config.json')
    
    assert os.path.exists(dir), "Cannot find {root_dir}"
    assert os.path.isfile(config_path), f"Config file {config_path} not found"
    assert name is not None, f"Name {name} is not valid"

    checkp_data = getLastCheckPoint(dir, name, scale=scale, iter=iteration)
    
    validate_checkpoint_data(
        checkp_data, dir, scale, iteration, name)

    modelConfig_path, pathModel, _ = checkp_data
    model_config = read_json(modelConfig_path)
    config = read_json(config_path)

    model = ProgressiveGAN(useGPU=True if GPU_is_available else False,
                      storeAVG=False,
                      **model_config)

    if scale is None or iteration is None:
        _, scale, iteration = parse_state_name(pathModel)

    print(f"Checkpoint found at scale {scale}, iter {iteration}")
    model.load(pathModel)

    model_name = get_filename(pathModel)
    return model, config, model_name

def validate_checkpoint_data(checkp_data, checkp_dir, scale, iter, name):
    if checkp_data is None:
        if scale is not None or iter is not None:
            raise FileNotFoundError(f"No checkpoint found for model {name} \
                at directory {checkp_dir} for scale {scale} at \
                iteration {iter}")
        
        raise FileNotFoundError(
            f"No checkpoint found for model {name} at directory {checkp_dir}")


# def get_dummy_nsynth_loader(config, nsynth_path):
#     from data.data_manager import AudioDataManager

#     data_path  = os.path.join(nsynth_path, "audio")
#     mdata_path = os.path.join(nsynth_path, "examples.json")

#     # load dymmy data_manager for post-processing
#     data_config = config["dataConfig"]

#     data_config["data_path"] = data_path
#     data_config["output_path"] = "/tmp"
#     data_config["loaderConfig"]["att_dict_path"] = mdata_path
#     data_config["loaderConfig"]["size"] = 500
#     dummy_dm = AudioDataManager(preprocess=True, **data_config)
#     return dummy_dm


def extract_save_rainbowgram(audio, path, name):

    fig = plt.figure(figsize=(10, 5))
    if type(audio) is torch.Tensor:
        audio = audio.numpy().reshape(-1).astype(float)
    audio = audio[:16000]
    rain = wave2rain(audio, sr=16000, stride=64, log_mag=True, clip=0.1)
    _ = rain2graph(rain)
    plt.xlabel('time (frames)')
    plt.ylabel('Frequency')
    plt.savefig(f'{path}/{name}.png')
    plt.close()

    # plt.show()

        
def saveAudioBatch(data, path, basename, sr=16000, overwrite=False):
    from librosa.util.utils import ParameterError
    # outdata = resizeAudioTensor(data, orig_sr, target_sr)
    # taudio.save(path, outdata, sample_rate=target_sr)
    try:
        for i, audio in enumerate(data):

            if type(audio) != np.ndarray:
                audio = np.array(audio, float)

            out_path = os.path.join(path, f'{basename}_{i}.wav')
            
            if not os.path.exists(out_path) or overwrite:
                write_wav(out_path, audio.astype(float), sr)
            else:
                print(f"saveAudioBatch: File {out_path} exists. Skipping...")
                continue
    except ParameterError as pe:
        print(pe)

class ResizeWrapper():
    def __init__(self, new_size):
        self.size = new_size
    def __call__(self, image):
        assert np.argmax(self.size) == np.argmax(image.shape[-2:]), \
            f"Resize dimensions mismatch, Target shape {self.size} \
                != image shape {image.shape}"
        if type(image) is not np.ndarray:
            image = image.numpy()
        out = interpolate(torch.from_numpy(image).unsqueeze(0), size=self.size).squeeze(0)
        return out