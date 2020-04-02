import os

from datetime import datetime
from .generation_tests import *

from utils.utils import mkdir_in_path, load_model_checkp, saveAudioBatch
from .generation_tests import StyleGEvaluationManager
from data.preprocessing import AudioProcessor


def generate(parser):
    args = parser.parse_args()

    model, config, model_name = load_model_checkp(**vars(args))
    latentDim = model.config.categoryVectorDim_G

    # We load a dummy data loader for post-processing
    postprocess = AudioProcessor(**config['transform_config']).get_postprocessor()

    # Create output evaluation dir
    output_dir = mkdir_in_path(args.dir, f"generation_tests")
    output_dir = mkdir_in_path(output_dir, model_name)
    output_dir = mkdir_in_path(output_dir, "random")
    output_dir = mkdir_in_path(output_dir, datetime.now().strftime('%Y-%m-%d %H:%M'))
    
    
    # Create evaluation manager
    eval_manager = StyleGEvaluationManager(model, n_gen=args.n_gen)

    gen_batch = eval_manager.test_random_generation()
    audio_out = map(postprocess, gen_batch)

    saveAudioBatch(audio_out,
                   path=output_dir,
                   basename='test_random', 
                   sr=config["transform_config"]["sample_rate"])
    print("FINISHED!\n")