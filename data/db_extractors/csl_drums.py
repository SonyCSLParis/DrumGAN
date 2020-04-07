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


__VERSION__ = "0.0.0"
MAX_N_FILES_IN_FOLDER = 10000

csl_keys = ['instrument', 'tags']
csl_instruments = ['kick', 'snare', 'hats']
csl_tags = ['lofi', 'techno', 'electronic', 'house', '808', '707', 'CL', '70s', \
            'trap', 'open', 'closed', 'cymbal' 'OP', 'clap', 'hiphop', 'clean', \
            'durty', 'short', 'long', 'punch']
csl_keys.sort()

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
    attributes = {'instrument': {'type': str(str), 'values': csl_instruments, 'count': {i: 0 for i in csl_instruments}}}

    i = 0
    for root, dirs, files in tqdm(os.walk(path)):
        if any(f.endswith('.wav') for f in files):
            inst_att = get_filename(os.path.dirname(root))
            files = list(filter(lambda x: x.endswith('.wav'), files))
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

    extraction_dict = get_base_db('csl-drums', __VERSION__)
    attribute_list = list(standard_desc['attributes'].keys())
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
            attribute_dict[att] = standard_desc['attributes'][att].copy()

        attribute_dict[att]['values'].sort()
        attribute_dict[att]['count'] = {str(k): 0 for k in attribute_dict[att]['values']}

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
