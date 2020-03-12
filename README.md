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
## Experiment config file

# Evaluation
You can run the evaluation metrics described in the paper: Pitch Inception Score (PIS), Instrument Inception Score (IIS), Pitch Kernel Inception Distance (PKID), Instrument Kernel Inception Distance (PKID) and the [Fréchet Audio Distance](https://arxiv.org/abs/1812.08466) (FAD).

* For computing Inception Scores run:
```
python eval.py <PIS or IIS> --fake <path_to_fake_data>
```

* For distance-like evaluation run:
```
python eval.py <PKID, IKID or FAD> --real_path <path_to_real_data> --fake <path_to_fake_data>
```

# Synthesizing audio with a model
```
python generate.py <random, scale, interpolation or from_midi> -d <path_to_model_root_folder> --nsynth-path <path_to_nsynth_data>
```
# Audio examples
[Here](https://sites.google.com/view/audio-synthesis-with-gans/p%C3%A1gina-principal) you can listen to audios synthesized with models trained on a variety of audio representations, includeing the raw audio waveform and several time-frequency representations.
# Notes
This repo is still a work in progress. Come later for more documentation and refactored code.
