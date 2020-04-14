from utils.utils import get_date
import hashlib


def get_base_db(name='default', version='0.0.0'):
    return {'name': name,
            'date': get_date(),
            'version': version,
            'extraction_config': '',
            'attributes': {},
            'data': []
        }

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