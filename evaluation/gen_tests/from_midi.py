import os

from datetime import datetime

from .generation_tests import *

from utils.utils import mkdir_in_path, load_model_checkp, saveAudioBatch
from .generation_tests import StyleGEvaluationManager
from data.preprocessing import AudioPreprocessor
from tqdm import trange
from mido import MidiFile
from librosa.output import write_wav


def interpolate_batch(x, y, steps):
    alpha = 0
    output = []
    for i in np.linspace(0., 1., steps, True):
        output.append(x*(1 - i) + y*i)
    return torch.stack(output)


def generate(parser):
    args = parser.parse_args()
    kwargs = vars(args)
    model, config, model_name = load_model_checkp(**kwargs)
    latentDim = model.config.categoryVectorDim_G
    overlap = kwargs.get('overlap', 0.77)
    batch_size = kwargs.get('batch_size', 50)

    # We load a dummy data loader for post-processing
    model_postpro = AudioPreprocessor(**config['transformConfig']).get_postprocessor()

    # Create output evaluation dir
    output_dir = mkdir_in_path(args.dir, f"generation_tests")
    output_dir = mkdir_in_path(output_dir, model_name)
    output_dir = mkdir_in_path(output_dir, "from_midi")
    output_dir = mkdir_in_path(output_dir, datetime.now().strftime('%Y-%m-%d %H:%M'))
    
    
    overlap_index = int(config['transformConfig']['audio_length']*overlap)

    print("Loading MIDI file")
    
    midi_file = MidiFile(args.midi)
    midi_name = os.path.basename(args.midi).split('.')[0]
    pitch_list = []
    pitch_range = config['loaderConfig']['pitch_range']
    pitch_cls_list = list(range(pitch_range[0], pitch_range[1] + 1))
    
    for i, track in enumerate(midi_file.tracks):
        for msg in track:

            if msg.type == "note_on":

                if msg.note in pitch_cls_list:
                    pitch_list.append(
                        pitch_cls_list.index(msg.note))
                else:
                    if msg.note > max(pitch_cls_list):
                        if msg.note - 12 in pitch_cls_list:
                            pitch_list.append(
                                pitch_cls_list.index(msg.note - 12))
                    if msg.note < min(pitch_cls_list):
                        if msg.note + 12 in pitch_cls_list:
                            pitch_list.append(
                                pitch_cls_list.index(msg.note + 12))

    output_audio = np.array([])
    pbar = trange(int(np.ceil(len(pitch_list)/batch_size)), desc="fake data IS loop")
    input_z, _ = model.buildNoiseData(batch_size, None, skipAtts=True)
    
    z = input_z[:, :-len(pitch_cls_list)].clone()
    z = interpolate_batch(z[0], z[1], steps=batch_size)
    n_interp = z.size(0)
    alpha = 0
    k = 0
    
    for j in pbar:
        input_labels = torch.LongTensor(pitch_list[j*batch_size: batch_size*(j+1)])
        input_z, _ = model.buildNoiseData(len(input_labels), inputLabels=input_labels.reshape(-1, 1), skipAtts=True)
        z_target = input_z[0, :-len(pitch_cls_list)].clone()
        input_z[:, :-len(pitch_cls_list)] = z.clone()[:len(input_labels)]
        gen_batch = model.test(input_z, getAvG=True)
        gen_raw = map(model_postpro, gen_batch)
        gen_raw = map(lambda x: np.array(x).astype(float), gen_raw)
        z = interpolate_batch(z[-1], z_target, batch_size)
        for i, g in enumerate(gen_raw):
            if i==0 and j == 0:
                output_audio = g
            else:
                output_audio = np.concatenate([output_audio, np.zeros(len(g) - overlap_index)])
                output_audio[-len(g):] += g
    # output_audio /= max(output_audio)
    # output_audio[output_audio > 1] = 1
    output_audio /= max(output_audio)
    write_wav(f'{output_dir}/{midi_name}_{datetime.today().strftime("%Y_%m_%d_%H")}.wav', output_audio, 16000)
