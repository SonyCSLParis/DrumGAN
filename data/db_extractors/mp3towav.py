import os
import argparse
import hashlib
import sys
import pickle
import requests

from utils.utils import get_date, mkdir_in_path, read_json, list_files_abs_path, get_filename, save_json
from random import shuffle
from tqdm import trange, tqdm

from .base_db import get_base_db, get_hash_dict

# from pydub import AudioSegment
# from mutagen.mp3 import MP3
from functools import partial

import ipdb
import subprocess
import multiprocessing
import signal
import resource


__VERSION__ = "0.0.0"
MAX_N_FILES_IN_FOLDER = 10000
criteria_keys = ['format1', 'format2', 'bitrate', 'parameters', 'size']
criteria_keys.sort()

def check_mp3_info(mp3_path):
    audio = MP3(mp3_path)
    return 

def get_mp3_pydub(criteria, path_mp3, file):
    filename = get_filename(file)
    out_file = os.path.join(path_mp3, filename + '.mp3')
    audio = AudioSegment.from_wav(file)
    audio.export(out_file, **criteria)
    return (file, out_file)

def get_mp3(criteria, path_mp3, file):
    filename = get_filename(file)
    out_file = os.path.join(path_mp3, filename + f".{criteria.get('format2', 'mp3')}")
    if not os.path.exists(out_file):
        subprocess.call(['ffmpeg', '-hide_banner', '-loglevel', 'panic', 
                        '-i', file, '-b:a', criteria.get('bitrate', '128k'), 
                        out_file])
    else:
        print(f"File {out_file} already exits")
    return (file, out_file)

def get_extractor_method(_format):
    return {
        '.mp3': get_mp3,
        'mp3.wav': get_mp3
    }[_format]


def extract(path_wav: str,
            path_mp3: str='', 
            dbname: str='mp3-to-wav', 
            criteria: dict={}):

    if criteria != {}:
        assert all(k in criteria_keys for k in criteria),\
            "Filter criteria not understood"
    if not os.path.exists(path_wav):
        print('NSynth folder not found')
        sys.exit(1)
    format1 = criteria.get('format1', 'wav')
    format2 = criteria.get('format2', 'mp3')

    extraction_hash = get_hash_dict(criteria)
    path_wav = path_wav.rstrip('/')
    root_dir = mkdir_in_path(os.path.dirname(path_wav), f'{dbname}_extractions')
    extraction_dir = mkdir_in_path(root_dir, str(extraction_hash))
    data_file = os.path.join(extraction_dir, 'data.pt')
    desc_file = os.path.join(extraction_dir, 'extraction.json')
    if os.path.exists(data_file):
        extraction_desc = read_json(desc_file)
        print(f"Loading {extraction_desc['name']}\n" \
              f"Version: {extraction_desc['version']}\n" \
              f"Date: {extraction_desc['date']}\n")
        
        return pickle.load(open(data_file, 'rb'))

    if not os.path.exists(path_mp3) or path_mp3 == '':
        path_wav.rstrip('/')
        path_mp3 = mkdir_in_path(os.path.dirname(path_wav), format2)

    description = get_base_db(dbname, __VERSION__)

    wav_files = list_files_abs_path(path_wav, format1)
    mp3_files = list_files_abs_path(path_mp3, format2)

    if len(mp3_files) > 0:
        wav_files = list(map(lambda x: \
            os.path.join(path_wav, get_filename(x.strip(format2)) + format1), mp3_files))
        wav_files = list(filter(lambda x: os.path.exists(x), wav_files))

        
    if len(wav_files) == 0:
        print('No wav files found!')
        sys.exit(1)

    if 'size' in criteria:
        size = criteria.pop('size')
    else:
        size = len(wav_files)

    extractor = get_extractor_method(format2)

    n_folders = 0
    data = []
    shuffle(wav_files)
    pbar = tqdm(wav_files[:size], desc='Reading files')
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (32768, rlimit[1]))
    p = multiprocessing.Pool(multiprocessing.cpu_count())
    data = list(p.map(partial(extractor, criteria, path_mp3), pbar))
    p.close()
    p.join()
    description['attributes'] = criteria
    description['output_file'] = data_file
    description['size'] = len(data)
    description['hash'] = extraction_hash

    with open(data_file, 'wb') as fp:
        pickle.dump((data, description), fp)
    save_json(description, desc_file)
    return data, description


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Nsynth database extractor')
    parser.add_argument('--wav', type=str, dest='path_wav',
                         help='Path to the nsynth root folder')

    parser.add_argument('--mp3', type=str, default='', dest='path_mp3',
                        help='Path to the nsynth root folder')

    parser.add_argument('-f', '--filter', help="Path to extraction configuration",
                        type=str, dest="filter_config", default=None)

    
    args = parser.parse_args()
    if args.filter_config != None:
        fconfig = read_json(args.filter_config)
    else:
        fconfig = {}
    fconfig = {
        'bitrate': "16k",
        'size': 10
    }
    extract(path_wav=args.path_wav,
            path_mp3=args.path_mp3,
            criteria=fconfig)
