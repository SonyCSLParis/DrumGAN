import os

from ..utils.utils import loadmodule, getLastCheckPoint, \
    parse_state_name, getNameAndPackage, read_json, GPU_is_available
from ..datasets.datamanager import AudioDataManager
from tools import get_filename

def load_model_checkp(dir, name=None, module='StyleGAN', iteration=None, scale=None, **kwargs):
    # Loading the model
    checkp_path = os.path.join(dir, name)
    config_path = os.path.join(checkp_path, f'{name}_config.json')
    
    assert os.path.exists(dir), "Cannot find {root_dir}"
    assert os.path.isfile(config_path), f"Config file {config_path} not found"
    assert name is not None, f"Name {name} is not valid"

    checkp_data = getLastCheckPoint(checkp_path, 
                                       name, 
                                       scale=scale, 
                                       iter=iteration)
    
    validate_checkpoint_data(
        checkp_data, checkp_path, scale, iteration, name)

    modelConfig_path, pathModel, _ = checkp_data
    model_config = read_json(modelConfig_path)
    config = read_json(config_path)

    modelPackage, modelName = getNameAndPackage(module)
    modelType = loadmodule(modelPackage, modelName)
    model = modelType(useGPU=True if GPU_is_available else False,
                      storeAVG=False,
                      **model_config)

    if scale is None or iteration is None:
        _, scale, iteration = parse_state_name(pathModel)

    print(f"Checkpoint found at scale {scale}, iter {iteration}")
    model.load(pathModel)

    model_name = get_filename(pathModel)
    return model, config, model_name, checkp_path

def validate_checkpoint_data(checkp_data, checkp_dir, scale, iter, name):
    if checkp_data is None:
        if scale is not None or iter is not None:
            raise FileNotFoundError(f"No checkpoint found for model {name} \
                at directory {checkp_dir} for scale {scale} at \
                iteration {iter}")
        
        raise FileNotFoundError(
            f"No checkpoint found for model {name} at directory {checkp_dir}")


def get_dummy_nsynth_loader(config, nsynth_path):

    data_path  = os.path.join(nsynth_path, "audio")
    mdata_path = os.path.join(nsynth_path, "examples.json")

    # load dymmy data_manager for post-processing
    data_config = config["dataConfig"]

    data_config["data_path"] = data_path
    data_config["output_path"] = "/tmp"
    data_config["loaderConfig"]["att_dict_path"] = mdata_path
    data_config["loaderConfig"]["size"] = 1000
    dummy_dm = AudioDataManager(preprocess=True,
                                **data_config)
    return dummy_dm


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
