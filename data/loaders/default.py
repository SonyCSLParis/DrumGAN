import os
import argparse
import hashlib
import sys
import pickle
import requests

from utils.utils import get_date, mkdir_in_path, read_json, list_files_abs_path, get_filename, save_json, walk_files_abs_path
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

criteria_keys = ['size', 'format']

def extract(path: str,
            dbname: str='default', 
            criteria: dict={}):

    if criteria != {}:
        assert all(k in criteria_keys for k in criteria),\
            "Filter criteria not understood"
    if not os.path.exists(path):
        print('Folder not found')
        sys.exit(1)

    _format = criteria.get('format', 'wav')

    extraction_hash = get_hash_dict(criteria)
    path = path.rstrip('/')
    root_dir = mkdir_in_path(os.path.dirname(path), f'{dbname}_extractions')
    extraction_dir = mkdir_in_path(root_dir, str(extraction_hash))
    data_file = os.path.join(extraction_dir, 'data.pt')
    desc_file = os.path.join(extraction_dir, 'extraction.json')
    if os.path.exists(data_file):
        extraction_desc = read_json(desc_file)
        print(f"Loading {extraction_desc['name']}\n" \
              f"Version: {extraction_desc['version']}\n" \
              f"Date: {extraction_desc['date']}\n")
        
        return pickle.load(open(data_file, 'rb'))

    description = get_base_db(dbname, __VERSION__)

    wav_files = walk_files_abs_path(path, _format)

    if len(wav_files) == 0:
        print('No wav files found!')
        sys.exit(1)

    size = criteria.get('size', len(wav_files))
    data = wav_files[:size]

    description['attributes'] = criteria
    description['output_file'] = data_file
    description['size'] = len(data)
    description['hash'] = extraction_hash

    with open(data_file, 'wb') as fp:
        pickle.dump((data, description), fp)
    save_json(description, desc_file)
    return data, description


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Default extractor')
    parser.add_argument('--path', type=str, dest='path',
                         help='Path to the nsynth root folder')

    parser.add_argument('-f', '--filter', help="Path to extraction configuration",
                        type=str, dest="filter_config", default=None)

    
    args = parser.parse_args()
    if args.filter_config != None:
        fconfig = read_json(args.filter_config)
    else:
        fconfig = {}
    fconfig = {
        'size': 10,
        'format': 'wav'
    }
    extract(path=args.path,
            criteria=fconfig)
