from utils.utils import get_date

def get_base_db(name='default', version='0.0.0'):
    return {'name': name,
            'date': get_date(),
            'version': version,
            'extraction_config': '',
            'attributes': {},
            'data': []
        }