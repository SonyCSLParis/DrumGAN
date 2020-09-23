# DrumGAN: Synthesis of Drum Sounds With Timbral Feature Conditioning Using Generative Adversarial Networks
This repo contains code for running DrumGAN, a Generative Adversarial Network that synthesizes drum sounds offering control over prcetpual features. You can find details about the specific architecture and the experiment in our ISMIR [paper](). Some of the codes are borrowed from [Facebook's GAN zoo repo](https://github.com/facebookresearch/pytorch_GAN_zoo).

# Notes
THIS REPO IS NOT UP TO DATE YET! Please, come back later. Sorry for the inconvenience.

# Install
1) Install requirements:

```
pip install -r requirements.txt
```
2) In order to compute the Fréchet Audio Distance (FAD) download and install google AI repo following the instructions [here](https://github.com/google-research/google-research/tree/master/frechet_audio_distance)

# The dataset
We train our model on a private, non-publicly available dataset containing 300k sounds of drum sounds equally distributed across kicks, snares and cymbals. This repo contains code for training a model on your own data. You will have to create a data loader, specific to the structure of your own dataset. 
# Training a new model
Train a new model from the module's root folder by executing:
```
python train.py $ARCH -c $PATH/TO/CONFIG/FILE
```
Available architectures:
* [PGAN](https://arxiv.org/abs/1710.10196)
## Example of config file:
The experiments are defined in a configuration file with JSON format.
```
{
    "name": "mag-if_test_config",
    "comments": "dummy configuration",
    "output_path": "/path/to/output/folder",
    "loader_config": {
        "dbname": "nsynth",
        "data_path": "/path/to/nsynth/audio/folder",
        "attribute_file": "/path/to/nsynth/examples.json",
        "filter_attributes": {
            "instrument_family_str": ["brass", "guitar", "mallet", "keyboard"],
            "instrument_source_str": ["acoustic"]
        },
        "shuffle": true,
        "attributes": ["pitch", "instrument_family_str"],
        "balance_att": "instrument_family_str",
        "pitch_range": [44, 70],
        "load_metadata": true,
        "size": 1000
    },
        
    "transform_config": {
        "transform": "specgrams",
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
    "model_config": {
        "formatLayerType": "default",
        "ac_gan": true,
        "downSamplingFactor": [
            [16, 16],
            [8, 8],
            [4, 4],
            [2, 2],
            [1, 1]
        ],
        "maxIterAtScale": [
            50,
            50,
            50,
            50,
            50
        ],
        "alphaJumpMode": "linear",
        "alphaNJumps": [
            600,
            600,
            600,
            600,
            1200
        ],
        "alphaSizeJumps": [
            32,
            32,
            32,
            32,
            32
        ],
        "transposed": false,
        "depthScales": [
            5,
            5,
            5,
            5,
            5
        ],
        "miniBatchSize": [
            2,
            2,
            2,
            2,
            2
        ],
        "dimLatentVector": 2,
        "perChannelNormalization": true,
        "lossMode": "WGANGP",
        "lambdaGP": 10.0,
        "leakyness": 0.02,
        "miniBatchStdDev": true,
        "baseLearningRate": 0.0006,
        "dimOutput": 1,
        "weightConditionG": 10.0,
        "weightConditionD": 10.0,
        "attribKeysOrder": {
            "pitch": 0,
            "instrument_family": 1
        },
        "startScale": 0,
        "skipAttDfake": []
    }
}

```

# Evaluation
You can run the evaluation metrics described in the paper: Inception Score (IS), Kernel Inception Distance (KID), and the [Fréchet Audio Distance](https://arxiv.org/abs/1812.08466) (FAD).

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
python generate.py <random, scale, radial_interpolation, spherical_interpolation, or from_midi> -d <path_to_model_root_folder>
```
# Audio examples
[Here](https://sites.google.com/view/drumgan) you can listen to audios synthesized with DrumGAN under different conditonal settings.

