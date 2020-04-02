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

import ipdb


__VERSION__ = "0.0.0"
MAX_N_FILES_IN_FOLDER = 10000

nsynth_keys = ['instrument_family_str', 'instrument', 'instrument_source_str', \
               'pitch', 'qualities_str', 'velocity']
from_nsynth_keys = {
    'instrument_family_str': 'instrument',
    'instrument': 'instrument_id',
    'instrument_source_str': 'instrument_type',
    'pitch': 'pitch',
    'qualities_str': 'properties',
    'velocity': 'velocity'

}
nsynth_train_url = 'http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz'
nsynth_valid_url = 'http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz'
nsynth_test_url = 'http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz'


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

def get_standard_format(path: str, dbname='nsynth'):
    nsynth_description = get_base_db('nsynth', __VERSION__)
    description_file = os.path.join(path, f'{dbname}.json')
    if os.path.exists(description_file):
        return read_json(description_file)

    nsynth_metadata = os.path.join(path, 'examples.json')
    nsynth_audio    = os.path.join(path, 'audio')

    root_dir = mkdir_in_path(path, f'nsynth_standard')

    extraction_config = os.path.join(root_dir, 'extraction_config.json')
    if os.path.exists(extraction_config):
        print("Extraction configuration exists. Loading...")
        return read_json(extraction_config)

    metadata = read_json(nsynth_metadata)
    nsynth_files = list_files_abs_path(nsynth_audio, '.wav')

    nsynth_description['total_size'] = len(nsynth_files)
    attributes = {}
    n_folders = 0

    pbar = tqdm(enumerate(nsynth_files), desc='Reading files')
    for i, file in pbar:
        if i % MAX_N_FILES_IN_FOLDER == 0:
            n_folders += 1
            output_dir = mkdir_in_path(root_dir, f'folder_{n_folders}')

        filename = get_filename(file)
        output_file = os.path.join(output_dir, filename + '.json')
        nsynth_description['data'].append(output_file)
        if os.path.exists(output_file):
            print(f'File {output_file} exists, skipping...')
        item = metadata[filename]
        out_item = {
            'path': file,
            'attributes': {}
        }
        for att in nsynth_keys:

            my_att = from_nsynth_keys[att]
            if my_att not in attributes:
                attributes[my_att] = {
                    'values': [],
                    'count': {}
                }
            if type(item[att]) in [int, str]:
                if item[att] not in attributes[my_att]['values']:
                    attributes[my_att]['values'].append(item[att])
                    attributes[my_att]['count'][str(item[att])] = 0
                attributes[my_att]['count'][str(item[att])] += 1
            if att == 'qualities_str':
                for q in item[att]:
                    if q not in attributes[my_att]['values']:
                        attributes[my_att]['values'].append(q)
                        attributes[my_att]['count'][str(q)] = 0
                    attributes[my_att]['count'][str(q)] += 1

            out_item['attributes'][my_att] = item[att]
        save_json(out_item, output_file)

    nsynth_description['attributes'] = attributes
    save_json(nsynth_description, description_file)
    return nsynth_description


def extract(path: str, criteria: dict={}, download: bool=False):
    criteria_keys = ['filter', 'balance', 'attributes', 'size']
    if criteria != {}:
        assert all(k in criteria_keys for k in criteria),\
            "Filter criteria not understood"
    if download:
        # downloading
        nsynth_dir = get_filename(path)
        nsynth_tar = requests.get(nsynth_train_url)
        with open(os.path.join(path, 'nsynth.tar.gz'), 'wb') as file:
            file.write(nsynth_tar.content)
            file.close()
    elif not os.path.exists(path):
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

    nsynth_standard_desc = get_standard_format(path)

    extraction_dict = get_base_db('nsynth', __VERSION__)
    attribute_list = list(nsynth_standard_desc['attributes'].keys())
    out_attributes = criteria.get('attributes', attribute_list)
    out_attributes.sort()

    # get database attribute values and counts 
    # given the filtering criteria
    attribute_dict = {att: {'values': [], 'count': {}} for att in out_attributes} 
    for att in attribute_dict.keys():
        if att in criteria.get('filter', {}).keys(): 
            if att in ['pitch', 'instrument_id']:

                attribute_dict[att]['values'] = list(range(*criteria['filter'][att]))
                
            else:
                criteria['filter'][att].sort()
                attribute_dict[att]['values'] = criteria['filter'][att]
        else:
            attribute_dict[att] = nsynth_standard_desc['attributes'][att].copy()

        attribute_dict[att]['values'].sort()
        attribute_dict[att]['count'] = {str(k): 0 for k in attribute_dict[att]['values']}

    size = criteria.get('size', nsynth_standard_desc['total_size'])
    balance = False
    if 'balance' in criteria:
        balance = True
        b_atts = criteria['balance']

        for b_att in b_atts:
            count = []
            for v in attribute_dict[b_att]['values']:
                
                count.append(nsynth_standard_desc['attributes'][b_att]['count'][str(v)])
            n_vals = len(count)
            size = min(size, n_vals * min(count))

    data = []
    metadata = []
    pbar = tqdm(nsynth_standard_desc['data'])

    for file in pbar:
        item = read_json(file)

        item_atts = item['attributes']
        item_path = item['path']

        # skip files that do not comply with
        # filtered attribute criteria
        skip = False
        for att, val in item_atts.items():
        # for att in out_attributes:
            # val = item_atts[att]
            if att in criteria.get('filter', {}):
                if att not in ['pitch', 'instrument_id']:
                    if val not in criteria['filter'][att]:
                        skip = True
                        break
            if att not in attribute_dict:
                continue

            if type(val) is list:
                if any(v in attribute_dict[att]['values'] for v in val) or val == []:
                    continue

            if val not in attribute_dict[att]['values']: 
                skip = True
                break
        if skip: continue

        # check balance of attributes
        if balance:
            for b_att in b_atts:
                val = item_atts[b_att]
                bsize = size / len(attribute_dict[b_att]['values'])
                if attribute_dict[b_att]['count'][str(val)] >= bsize:
                    skip = True
            if skip:
                continue
        # store attribute index in list
        data_item = []
        for att in out_attributes:
            val = item_atts[att]
            # if attribute is multi-label (n out of m)
            if type(val) is list:
                if all(isinstance(n, str) for n in val):
                    # we now consider binary attributes (1 or 0)
                    data_val = [0] * len(attribute_dict[att]['values'])
                    for v in val:
                        if v in attribute_dict[att]['values']:
                            idx = attribute_dict[att]['values'].index(v)
                            attribute_dict[att]['count'][str(v)] += 1
                            data_val[idx] = 1
                        else:
                            continue
                # TODO: consider float values (audioset) 
                # --> counts and value tracking makes no sense
                elif all(isinstance(n, float) for n in val):
                    pass
            else:
                idx = attribute_dict[att]['values'].index(val)
                attribute_dict[att]['count'][str(val)] += 1
                data_val = idx
            data_item.append(data_val)
        if skip: continue
        data.append(item_path)
        metadata.append(data_item)
        extraction_dict['data'].append(file)
        if len(data) >= size:
            pbar.close()
            break
        
    extraction_dict['attributes'] = attribute_dict
    extraction_dict['output_file'] = data_file
    extraction_dict['size'] = len(data)
    extraction_dict['hash'] = extraction_hash
    with open(data_file, 'wb') as fp:
        pickle.dump((data, metadata, extraction_dict), fp)
    save_json(extraction_dict, desc_file)
    return data, metadata, extraction_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Nsynth database extractor')
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
