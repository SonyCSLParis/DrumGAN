import sys
import argparse
import sys
from datetime import datetime

from torch.backends import cudnn

from data.loaders import get_data_loader
from data.preprocessing import AudioProcessor
from gans import ProgressiveGANTrainer
from utils.config import update_parser_with_config, \
    get_config_override_from_parser
from utils.utils import *

# get rid of the librosa warning when loading mp3s
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore") # Change the filter in this process
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

from datetime import datetime
from visualization import getVisualizer


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Deep-Audio-GenLib training script')
    parser.add_argument('architecture', type=str, default='PGAN',
                         help='Name of the architecture to launch, available models are\
                         PGAN and PPGAN. To get all possible option for a model\
                         please run train.py $MODEL_NAME')
    parser.add_argument('-n', '--name', help="Model's name",
                        type=str, dest="name", 
                        default=f"default_{datetime.now().strftime('%y_%m_%d')}")
    parser.add_argument('-d', '--dataset',type=str, dest="dataset", default=f"nsynth", 
                        help="Dataset name. Availabel: nsynth, mtg-drums, csl-drums, youtube-pianos")
    
    parser.add_argument('-o', '--output-path', help='Output directory',
                        type=str, dest="output_path", default='output_networks')
    parser.add_argument('-c', '--config', help="Path to configuration file",
                        type=str, dest="config")
    parser.add_argument('-s', '--save_iter', help="If it applies, frequence at\
                        which a checkpoint should be saved. In the case of a\
                        evaluation test, iteration to work on.",
                        type=int, dest="save_i", default=1000)
    parser.add_argument('-e', '--eval_iter', help="If it applies, frequence at\
                        which evaluation is run", 
                        type=int, dest="eval_i", default=-1)
    parser.add_argument('-l', '--loss_iter', help="If it applies, frequence at\
                        which a checkpoint should be saved. In the case of a\
                        evaluation test, iteration to work on.",
                        type=int, dest="loss_i", default=5000)
    parser.add_argument('--scale', dest="scale", default=0, 
                        help="If checkpoints found, start at scale")
    parser.add_argument('--iter', help="If chekpoints found, iteration at which to continue")
    parser.add_argument('-v', '--partitionValue', help="Partition's value",
                        type=str, dest="partition_value")
    parser.add_argument('--seed', dest='seed', action='store_true', help="Partition's value")
    parser.add_argument('--n_samples', type=int, default=10, help="Partition's value")
    parser.add_argument('--restart', action='store_true',
                        help=' If a checkpoint is detected, do not try to load it')
    parser.add_argument('--no-visdom', action='store_true', dest='no_visdom',
                        help='Deactivate visdom visualization')

    import resource

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (32768, rlimit[1]))

    #torch.autograd.set_detect_anomaly(True)
    cudnn.benchmark = True

    # Parse command line args
    args, unknown = parser.parse_known_args()
    # Initialize random seed
    init_seed(args.seed)
    # Build the output directory if necessary
    checkexists_mkdir(args.output_path)
    # Add overrides to the parser: changes to the model configuration can be
    # done via the command line
    parser = update_parser_with_config(parser, ProgressiveGANTrainer._defaultConfig)
    kwargs = vars(parser.parse_args())
    config_override = get_config_override_from_parser(kwargs, ProgressiveGANTrainer._defaultConfig)

    if kwargs['overrides']:
        parser.print_help()
        sys.exit()

    # configuration file path
    config = load_config_file(args.config)
    config['arch'] = args.architecture

    # Retrieve the model we want to launch
    print(f"Loading traines for {args.architecture}")
    trainerModule = get_trainer(args.architecture)

    # model config
    model_config = config["model_config"]
    for item, val in config_override.items():
        model_config[item] = val

    # # data config
    # for item, val in config_override.items():
    #     data_config[item] = val

    exp_name = config.get("name", "default")
    checkpoint_dir = config["output_path"]
    checkpoint_dir = mkdir_in_path(checkpoint_dir, exp_name)
    # config["output_shapetput_path"] = checkpoint_dir

    # configure processor
    print("Data manager configuration")
    transform_config = config['transform_config']
    audio_processor = AudioProcessor(**transform_config)
    
    # configure loader
    loader_config = config['loader_config']
    dbname = loader_config.pop('dbname', args.dataset)

    loader_module = get_data_loader(dbname)

    loader = loader_module(dbname=dbname + '_' + transform_config['transform'],
                           output_path=checkpoint_dir, 
                           preprocessing=audio_processor,
                           **loader_config)

    print(f"Loading data. Found {len(loader)} instances")
    model_config['output_shape'] = audio_processor.get_output_shape()
    config["model_config"] = model_config

    # visualization
    vis_manager = \
    getVisualizer(transform_config['transform'])(
        output_path=checkpoint_dir,
        env=exp_name,
        sampleRate=transform_config.get('sample_rate', 16000),
        no_visdom=args.no_visdom)


    # save config file
    save_json(config, os.path.join(checkpoint_dir, f'{exp_name}_config.json'))

    GANTrainer = trainerModule(
        model_name=exp_name,
        gpu=GPU_is_available(),
        loader=loader,
        loss_plot_i=args.loss_i,
        eval_i=args.eval_i,
        checkpoint_dir=checkpoint_dir,
        save_iter=args.save_i,
        n_samples=args.n_samples,
        config=model_config,
        vis_manager=vis_manager)

    # load checkpoint
    print("Search and load last checkpoint")
    checkpoint_state = getLastCheckPoint(checkpoint_dir, exp_name)
    # If a checkpoint is found, load it
    if not args.restart and checkpoint_state is not None:
        train_config, model_path, tmp_data_path = checkpoint_state
        GANTrainer.load_saved_training(model_path, train_config, tmp_data_path)

    GANTrainer.train()
