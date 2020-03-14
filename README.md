# Comparing Representations for Audio Synthesis using GANs
This repo contains code for comparing audio representations on the task of audio synthesis with Generative Adversarial Networks (GAN). Some of the codes are borrowed from [Facebook's GAN zoo repo](https://github.com/facebookresearch/pytorch_GAN_zoo).
# Install
1) Install requirements:

```
pip install -r requirements.txt
```
2) Download and install Python implementation of [NSGT transform](https://github.com/grrrr/nsgt) 
3) In order to compute the Fréchet Audio Distance (FAD) download and install google AI repo following the instructions [here](https://github.com/google-research/google-research/tree/master/frechet_audio_distance)

# The dataset
We use a subset of the [Nsynth datasaet](https://magenta.tensorflow.org/datasets/nsynth) as described in the paper accompanying this github repository.
# Training a new model
Train a new model from the module's root folder by executing:
```
python train.py -c <path-to-configuration-file>
```
## Example of config file:
The experiments are defined in a configuration file with JSON format.
```
{
    "name": "mag-if_test_config",
    "comments": "dummy_configuration",
    "output_path": "output/path",
    "loaderConfig": {
        "data_path": "path/to/nsynth/audio/folder",
        "att_dict_path": "path/to/nsynth/metadata/file",
        "filter": ["acoustic"],
        "instrument_labels": ["brass", "mallet", "flute", "guitar"],
        "shuffle": false,
        "attribute_list": ["pitch"],
        "pitch_range": [40, 70],
        "load_metadata": true,
        "size": 1000
    },
        
    "transformConfig": {
		"transform": "mag-if",
        "fade_out": true,
        "fft_size": 1024,
        "win_size": 1024,
        "n_frames": 64,
        "hop_size": 256,
        "log": true,
        "ifreq": true,
        "sample_rate": 16000,
        "audio_length": 16000
    },
    "modelConfig": {
        "formatLayerType": "default",
        "ac_gan": true,
        "downSamplingFactor": [
            [16, 16],
            [8, 8],
            [4, 4],
            [2, 2],
            [1, 1]
        ],
        "imagefolderDataset": true,
        "maxIterAtScale": [50, 50, 50, 50, 50],
        "alphaJumpMode": "linear",
        "alphaNJumps": [600, 600, 600, 600, 1200],
        "alphaSizeJumps": [32, 32, 32, 32, 32],
        "transposed": false,
        "depthScales": [5, 5, 5, 5, 5],
        "miniBatchSize": [2, 2, 2, 2, 2],
        "dimLatentVector": 2,
        "perChannelNormalization": true,
        "lossMode": "WGANGP",
        "lambdaGP": 10.0,
        "leakyness": 0.02,
        "miniBatchStdDev": true,
        "baseLearningRate": 0.0006,
        "dimOutput": 2,
        "weightConditionG": 10.0,
        "weightConditionD": 10.0,
        "attribKeysOrder": {
            "pitch": 0
        },
        "startScale": 0,
        "skipAttDfake": []
    }
}

```

# Evaluation
You can run the evaluation metrics described in the paper: Pitch Inception Score (PIS), Instrument Inception Score (IIS), Pitch Kernel Inception Distance (PKID), Instrument Kernel Inception Distance (PKID) and the [Fréchet Audio Distance](https://arxiv.org/abs/1812.08466) (FAD).

* For computing Inception Scores run:
```
python eval.py <pis or iis> --fake <path_to_fake_data> -d <output_path>
```

* For distance-like evaluation run:
```
python eval.py <pkid, ikid or fad> --real <path_to_real_data> --fake <path_to_fake_data> -d <output_path>
```

# Synthesizing audio with a model
```
python generate.py <random, scale, interpolation or from_midi> -d <path_to_model_root_folder>
```
# Audio examples
[Here](https://sites.google.com/view/audio-synthesis-with-gans/p%C3%A1gina-principal) you can listen to audios synthesized with models trained on a variety of audio representations, includeing the raw audio waveform and several time-frequency representations.
# Notes
This repo is still a work in progress. Come later for more documentation and refactored code.
