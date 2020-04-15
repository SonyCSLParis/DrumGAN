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

import ipdb
import numpy as np

__VERSION__ = "0.0.0"
MAX_N_FILES_IN_FOLDER = 10000

csl_keys = ['instrument', 'tags', 'audio-commons']
csl_keys.sort()

csl_instruments = ['kick', 'snare', 'hats']
csl_instruments.sort()

csl_tags = ['lofi', 'techno', 'electronic', 'house', '808', '707', 'CL', '70s', \
            'trap', 'open', 'closed', 'cymbal' 'OP', 'clap', 'hiphop', 'clean', \
            'durty', 'short', 'long', 'punch']
csl_tags.sort()

audio_commons_keys = ['duration', 'loudness', 'dynamic_range', \
            'temporal_centroid', 'log_attack_time', \
            'hardness', 'depth', 'brightness', 'roughness',\
            'boominess', 'warmth', 'sharpness']
audio_commons_keys.sort()

def get_standard_format(path: str, dbname='csl-drums'):
    description = get_base_db(dbname, __VERSION__)
    description_file = os.path.join(path, f'{dbname}.json')
    if os.path.exists(description_file):
        return read_json(description_file)

    root_dir = mkdir_in_path(path, f'csl_drums_standard')

    extraction_config = os.path.join(root_dir, 'extraction_config.json')
    if os.path.exists(extraction_config):
        print("Extraction configuration exists. Loading...")
        return read_json(extraction_config)

    n_folders = 0
    attributes = {'instrument': {
                    'type': str(str),
                    'loss': 'xentropy', 
                    'values': csl_instruments, 
                    'count': {i: 0 for i in csl_instruments}}}

    i = 0
    for root, dirs, files in tqdm(os.walk(path)):
        if any(f.endswith('.wav') for f in files):
            inst_att = get_filename(os.path.dirname(root))
            files = list(filter(lambda x: x.endswith('.wav'), files))
            files = list(filter(lambda x: not x.startswith('._'), files))
            # filter files that have _analysis.json
            files = list(filter(lambda x: os.path.exists(os.path.join(root, get_filename(x) + '_analysis.json')), files))
            attributes['instrument']['count'][inst_att] += len(files)

            # pbar = tqdm(files, desc=f'Reading {inst_att} files')
            for file in files:
                file = os.path.join(root, file)
                if i % MAX_N_FILES_IN_FOLDER == 0:
                    n_folders += 1
                    output_dir = mkdir_in_path(root_dir, f'folder_{n_folders}')

                filename = get_filename(file)
                
                output_file = os.path.join(output_dir, filename + '.json')
                description['data'].append(output_file)

                out_item = {
                    'path': file,
                    'attributes': {}
                }
                for att in csl_keys:

                    if att == 'audio-commons':
                        if att not in attributes:
                            attributes[att] = {
                                'values': audio_commons_keys,
                                'type': str(float),
                                'loss': 'mse',
                                'max': {a: -1000.0 for a in audio_commons_keys},
                                'mean': {a: 0.0 for a in audio_commons_keys},
                                'min': {a: 1000.0 for a in audio_commons_keys},
                                'var': {a: 0.0 for a in audio_commons_keys}
                            }
                        ac_file = os.path.join(root, filename + '_analysis.json')
                        assert os.path.exists(ac_file), f"File {ac_file} does not exist"
                        ac_atts = read_json(ac_file)
                        out_item['attributes'][att] = {}
                        for ac_att in audio_commons_keys:
                            if ac_att not in ac_atts: continue
                            acval = ac_atts[ac_att]
                            out_item['attributes'][att][ac_att] = acval
                            if acval > attributes[att]['max'][ac_att]:
                                attributes[att]['max'][ac_att] = acval
                            if acval < attributes[att]['min'][ac_att]:
                                attributes[att]['min'][ac_att] = acval
                            attributes[att]['mean'][ac_att] += acval
                    
                    if att not in attributes:
                        attributes[att] = {
                            'values': [],
                            'count': {}
                        }
                    if att == 'tags':
                        import re
                        tags = []
                        tag = re.sub("\d+", " ", filename)
                        tag = re.sub('^[^A-Za-z]*', '', tag)
                        tag = tag.split('_')
                        for t in tag:
                            tags += re.findall(r"[\w']+", t)

                        for t in tags:
                            if t not in attributes[att]['values']:
                                attributes[att]['values'].append(t)
                                attributes[att]['count'][t] = 0
                            attributes[att]['count'][t] += 1
                        out_item['attributes'][att] = tags
                    elif att == 'instrument':
                        out_item['attributes'][att] = inst_att
                i+=1
                save_json(out_item, output_file)
    for ac in audio_commons_keys:
        attributes['audio-commons']['mean'][ac] /= i
    description['attributes'] = attributes
    description['total_size'] = len(description['data'])
    save_json(description, description_file)
    return description


def extract(path: str, criteria: dict={}, download: bool=False):
    criteria_keys = ['balance', 'attributes', 'size', 'filter']
    criteria_keys.sort()

    if criteria != {}:
        assert all(k in criteria_keys for k in criteria),\
            "Filter criteria not understood"

    if not os.path.exists(path):
        print('CSL folder not found')
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

    extraction_dict = get_base_db('csl-drums', __VERSION__)
    attribute_list = list(standard_desc['attributes'].keys())
    out_attributes = criteria.get('attributes', attribute_list)
    out_attributes.sort()

    # get database attribute values and counts 
    # given the filtering criteria
    attribute_dict = {} 
    for att in out_attributes:
        if att not in attribute_dict:
            if att == 'audio-commons':
                attribute_dict[att] = {
                    'values': [],
                    'type': str(float),
                    'loss': 'mse',
                    'max': {},
                    'min': {},
                    'mean': {},
                    'var': {}
                }
            else:
                attribute_dict[att] = {
                    'values': [],
                    'type': str(str),
                    'loss': 'xentropy',
                    'count': {}
                }
        if att in criteria.get('filter', {}).keys():
            criteria['filter'][att].sort()

            attribute_dict[att]['values'] = criteria['filter'][att]
            if att == 'audio-commons':
                for ac_att in criteria['filter'][att]:
                    attribute_dict[att]['max'][ac_att] = standard_desc['attributes'][att]['max'][ac_att]
                    attribute_dict[att]['min'][ac_att] = standard_desc['attributes'][att]['min'][ac_att]
                    attribute_dict[att]['mean'][ac_att] = standard_desc['attributes'][att]['mean'][ac_att]
                    attribute_dict[att]['var'][ac_att] = 0.0
        else:
            attribute_dict[att] = standard_desc['attributes'][att].copy()

        attribute_dict[att]['values'].sort()
        attribute_dict[att]['count'] = {str(k): 0 for k in attribute_dict[att]['values']}

    # get absolute max for normalization value
    if 'audio-commons' in attribute_dict:
        for i, att in enumerate(attribute_dict['audio-commons']['values']):
            if i == 0:
                max_norm_val = attribute_dict['audio-commons']['max'][att]
                min_norm_val = attribute_dict['audio-commons']['min'][att]
            else:
                if attribute_dict['audio-commons']['max'][att] > max_norm_val:
                    max_norm_val = attribute_dict['audio-commons']['max'][att]
                if attribute_dict['audio-commons']['min'][att] < min_norm_val:
                    min_norm_val = attribute_dict['audio-commons']['min'][att]
    

    size = criteria.get('size', standard_desc['total_size'])
    balance = False
    if 'balance' in criteria:
        balance = True
        b_atts = criteria['balance']

        for b_att in b_atts:
            count = []
            for v in attribute_dict[b_att]['values']:
                count.append(standard_desc['attributes'][b_att]['count'][str(v)])
            n_vals = len(count)
            size = min(size, n_vals * min(count))

    data = []
    metadata = []
    pbar = tqdm(standard_desc['data'])

    for file in pbar:
        item = read_json(file)

        item_atts = item['attributes']
        item_path = item['path']

        # skip files that do not comply with
        # filtered attribute criteria
        skip = False
        for att, val in item_atts.items():
            if att not in attribute_dict:
                continue
            if att == 'audio-commons':
                for ac_att in attribute_dict[att]['values']:
                    if ac_att not in val.keys():
                        skip = True
                        break
                    if np.isnan(val[ac_att]):
                        print(f"NaN value found in file {file} and att {ac_att}, skipping...")
                        skip = True
                        break
            elif att in criteria.get('filter', {}):
                if val not in criteria['filter'][att]:
                    skip = True
                    break
            else:
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
            if att == 'audio-commons':
                
                for ac_att in attribute_dict[att]['values']:
                    acval = (val[ac_att] - min_norm_val) / (max_norm_val - min_norm_val)
                    data_item += [acval]
                    attribute_dict[att]['var'][ac_att] += \
                        (attribute_dict[att]['mean'][ac_att] - val[ac_att])**2
            else:
                idx = attribute_dict[att]['values'].index(val)
                attribute_dict[att]['count'][str(val)] += 1
                data_item += [idx]
            # data_item.append(data_val)
        if skip: continue

        data.append(item_path)
        metadata.append(data_item)
        extraction_dict['data'].append(file)
        if len(data) >= size:
            pbar.close()
            break
        
    # compute std:
    
    if 'audio-commons' in attribute_dict:
        for att in attribute_dict['audio-commons']['values']:
            attribute_dict['audio-commons']['var'][att] /= len(data)
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
