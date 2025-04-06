import logging
import time
import warnings
from typing import Optional

import requests
from pyanimeinfo.myanimelist import JikanV4Client
from requests import HTTPError

from .session import get_requests_session


def get_items_from_myanimelist(title: str, session: Optional[requests.Session] = None):
    logging.info('Search information from myanimelist ...')
    session = session or get_requests_session()
    jikan_client = JikanV4Client(session=session)
    while True:
        try:
            jikan_items = jikan_client.search_anime(query=title)
        except HTTPError as err:
            if err.response.status_code == 429:
                warnings.warn(f'429 error detected: {err!r}, wait for some seconds ...')
                time.sleep(5.0)
            else:
                raise
        else:
            break

    type_map = {'tv': 0, 'movie': 1, 'ova': 2, 'ona': 2}
    retval, exist_mal_ids = [], set()
    for i, pyaitem in enumerate(jikan_items, start=1):
        if not pyaitem['type'] or pyaitem['type'].lower() not in type_map:
            continue
        if pyaitem['mal_id'] in exist_mal_ids:
            continue

        logging.info(f'Finding result #{i}, id: {pyaitem["mal_id"]!r}, '
                     f'type: {pyaitem["type"]!r}, title: {pyaitem["title"]!r}.')
        retval.append(pyaitem)
        exist_mal_ids.add(pyaitem['mal_id'])

    return retval
