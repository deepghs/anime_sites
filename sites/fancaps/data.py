import json
from functools import lru_cache
from pprint import pprint

from hfutils.operate import get_hf_client


@lru_cache()
def _get_mappings():
    hf_client = get_hf_client()
    with open(hf_client.hf_hub_download(
            repo_id='deepghs/fancaps_index',
            repo_type='dataset',
            filename='bangumi.json'
    ), 'r') as f:
        data = json.load(f)
    return data


if __name__ == '__main__':
    pprint(_get_mappings())
