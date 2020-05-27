import os
import argparse
import hashlib
import sys
import pickle
import requests

from utils.utils import get_date, mkdir_in_path, read_json, list_files_abs_path, get_filename, save_json
from random import shuffle
from tqdm import trange, tqdm

from .base_db import get_base_db
from numpy import isnan
import ipdb


__VERSION__ = "0.0.0"
MAX_N_FILES_IN_FOLDER = 10000

mtg_keys = ['duration', 'loudness', 'dynamic_range', \
            'temporal_centroid', 'log_attack_time', \
            'hardness', 'depth', 'brightness', 'roughness',\
            'boominess', 'warmth', 'sharpness']
mtg_keys.sort()


def get_hash_dict(_dict: dict):
    keys = list(_dict.keys())
    keys.sort()
    hash_list = []
    for k in keys:
        # TODO: list here will break if the list is of objects
        if type(_dict[k]) in [list, str, int, float]:
            hash_list.append((k, _dict[k]))
        elif type(_dict[k]) is dict:
            hash_list.append(get_hash_dict(_dict[k]))
    return hashlib.sha1(str(hash_list).encode()).hexdigest()

def get_standard_format(path: str, dbname='mtg-drums'):
    base_description = get_base_db(dbname, __VERSION__)
    description_file = os.path.join(path, f'{dbname}.json')
    if os.path.exists(description_file):
        return read_json(description_file)

    metadata_dir = os.path.join(path, 'analysis')
    root_dir = mkdir_in_path(path, f'mtg_drums_standard')

    audio_files = list_files_abs_path(path, '.wav')

    base_description['total_size'] = len(audio_files)
    attributes = {}
    n_folders = 0

    pbar = tqdm(enumerate(audio_files), desc='Reading files')

    for i, file in pbar:
        if i % MAX_N_FILES_IN_FOLDER == 0:
            n_folders += 1
            output_dir = mkdir_in_path(root_dir, f'folder_{n_folders}')

        filename = get_filename(file)
        metadata_file = os.path.join(metadata_dir, filename + '_analysis.json')
        if not os.path.exists(metadata_file):
            print(f"File {file} does not have associated \
                    metadata file {metadata_file}")
            continue
        item = read_json(metadata_file)

        output_file = os.path.join(output_dir, filename + '.json')
        base_description['data'].append(output_file)

        out_item = {
            'path': file,
            'attributes': {}
        }
        for att in mtg_keys:
            if att not in item:
                print(f"Attribute {att} not in {file}")
                continue
            if isnan(item[att]):
                print(f"Encountered NaN value in {att} for file {file}")
                continue
            att_type = type(item[att])
            if att not in attributes:
                # if att == 'reverb':
                #     attributes[att] = {
                #         'type': str(att_type),
                #         'values': [],
                #         'count': {}
                #     }
                # else:
                attributes[att] = {
                    'type': str(att_type),
                    'loss': 'mse',
                    'values': [att],
                    'max': -1000.0,
                    'min': 10000.0,
                    'mean': 0.0,
                    'count': []
                }   
            # if att == 'reverb':
            #     if item[att] not in attributes[att]['values']:
            #         attributes[att]['values'].append(item[att])
            #         attributes[att]['count'][str(item[att])] = 0
            #     attributes[att]['count'][str(item[att])] += 1
            # else:

            if item[att] > attributes[att]['max']:
                attributes[att]['max'] = item[att]
            if item[att] < attributes[att]['min']:
                attributes[att]['min'] = item[att]
            attributes[att]['mean'] += item[att]

            out_item['attributes'][att] = item[att]
        save_json(out_item, output_file)
    for att in attributes:
        if attributes[att]['type'] == str(float):
            attributes[att]['mean'] /= i
    base_description['attributes'] = attributes
    save_json(base_description, description_file)
    return base_description


def extract(path: str, criteria: dict={}):
    criteria_keys = ['quantize', 'attributes', 'size']
    quantize = False
    norm = True
    criteria_keys.sort()
    if criteria != {}:
        assert all(k in criteria_keys for k in criteria),\
            "Filter criteria not understood"

    if not os.path.exists(path):
        print('NSynth folder not found')
        sys.exit(1)

    root_dir = mkdir_in_path(path, f'extractions')
    extraction_hash = get_hash_dict(criteria)

    extraction_dir = mkdir_in_path(root_dir, str(extraction_hash))
    data_file = os.path.join(extraction_dir, 'data.pt')
    desc_file = os.path.join(extraction_dir, 'extraction.json')

    if os.path.exists(data_file):
        extraction_desc = read_json(desc_file)

        print(f"Loading {extraction_desc['name']}\n" \
              f"Version: {extraction_desc['version']}\n" \
              f"Date: {extraction_desc['date']}\n")
        return pickle.load(open(data_file, 'rb'))

    standard_desc = get_standard_format(path)

    extraction_dict = get_base_db('mtg-drums', __VERSION__)
    attribute_list = list(standard_desc['attributes'].keys())
    out_attributes = criteria.get('attributes', attribute_list)
    out_attributes.sort()

    # get database attribute values and counts 
    # given the filtering criteria
    attribute_dict = {att: standard_desc['attributes'][att] for att in out_attributes} 

    size = criteria.get('size', standard_desc['total_size'])
    data = []
    metadata = []
    pbar = tqdm(standard_desc['data'])
    for file in pbar:
        item = read_json(file)

        item_atts = item['attributes']
        item_path = item['path']

        # skip files that do not comply with
        # filtered attribute criteria
        data_item = []
        skip=False
        for att in out_attributes:
            if att not in item_atts:
                skip=True
                break
            val = item_atts[att]
            if True:
                val = (val - attribute_dict[att]['min']) / (attribute_dict[att]['max'] - attribute_dict[att]['min'])
            if att not in attribute_dict:
                continue
            if 'std' not in attribute_dict[att]:
                attribute_dict[att]['var'] = 0.0
            attribute_dict[att]['var'] += (val - attribute_dict[att]['mean'])**2
            data_item.append(val)
        if skip:
            continue
        data.append(item_path)
        metadata.append(data_item)
        extraction_dict['data'].append(file)
        if len(data) >= size:
            pbar.close()
            break
    
    # compute std:
    for att in attribute_dict:
        attribute_dict[att]['var'] /= len(data)

    extraction_dict['attributes'] = attribute_dict
    extraction_dict['output_file'] = data_file
    extraction_dict['size'] = len(data)
    extraction_dict['hash'] = extraction_hash

    with open(data_file, 'wb') as fp:
        pickle.dump((data, metadata, extraction_dict), fp)
    save_json(extraction_dict, desc_file)
    return data, metadata, extraction_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MTG-Drums database extractor')
    parser.add_argument('nsynth_path', type=str,
                         help='Path to the nsynth root folder')
    
    parser.add_argument('-f', '--filter', help="Path to extraction configuration",
                        type=str, dest="filter_config", default=None)
    
    parser.add_argument('--download', action='store_true', 
                        help="Download nsynth?",
                        dest="download", default=False)
    
    args = parser.parse_args()
    if args.filter_config != None:
        fconfig = read_json(args.filter_config)
    else:
        fconfig = {}
    # fconfig = {
    #     'attributes': ['pitch'],
    #     'balance': ['pitch'],
    #     'filter': {
    #         'instrument': ['mallet'],
    #         'pitch': [30, 50]
    #     }
    # }
    extract(path=args.nsynth_path,
            criteria=fconfig,
            download=args.download)
