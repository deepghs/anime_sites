import datetime
from typing import Optional

import dateparser
import requests
from pyanimeinfo.myanimelist import JikanV4Client

from .info import get_info_from_subsplease
from ..utils import get_requests_session


def get_full_info_for_replace(page_url: str, mal_id: int, session: Optional[requests.Session] = None):
    session = session or get_requests_session()
    subs_info = get_info_from_subsplease(page_url, session=session)
    del subs_info['prompt']

    jikan_client = JikanV4Client(session=session)
    mal_info = jikan_client.get_anime_full(mal_id)

    min_timestamp = None
    for item in [*subs_info['batch'], *subs_info['episode']]:
        timestamp = dateparser.parse(item['release_date']).timestamp()
        if not min_timestamp or timestamp < min_timestamp:
            min_timestamp = timestamp
    year = datetime.datetime.fromtimestamp(min_timestamp).year if min_timestamp else None

    return {
        'mal_id': mal_info['mal_id'],
        'title': mal_info['title'],
        'reason': f'Admin specified {page_url!r} to mal #{mal_id}',
        'mal': mal_info,
        'year': year,
        'subsplease': subs_info,
    }
