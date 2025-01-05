import io
from typing import Optional
from urllib.parse import urljoin

import requests
from ditk import logging
from hbutils.system import urlsplit
from pyquery import PyQuery as pq

from ..utils import get_requests_session


def get_info_from_subsplease(page_url: str, session: Optional[requests.Session] = None):
    session = session or get_requests_session()
    logging.info(f'Accessing page {page_url!r} ...')
    resp = session.get(page_url)
    resp.raise_for_status()

    assert urlsplit(page_url).path_segments[1] == 'shows'
    page_id = urlsplit(page_url).path_segments[2]

    page = pq(resp.text)
    title = page('h1.entry-title').text().strip()
    synopsis = page('.series-syn').text().strip()
    cover_image_url = urljoin(resp.url, page('#site-sidebar img.img-center').attr('src')) \
        if page('#site-sidebar img.img-center').attr('src') else None

    with io.StringIO() as sf:
        print(page('div.entry-content').text().strip(), file=sf)
        print(f'', file=sf)

        sid_value = page('#show-release-table').attr('sid')
        if sid_value:
            logging.info(f'Getting release list of sid {sid_value!r} ...')
            r = session.get('https://subsplease.org/api/', params={
                'f': 'show', 'tz': 'Asia/Tokyo', 'sid': sid_value,
            })
            r.raise_for_status()

            batch = r.json().get('batch') or {}
            if batch:
                print('Batch', file=sf)
                print(f'', file=sf)
                for key, item in batch.items():
                    print(f'#{item["episode"]} - {key!r} - {item["release_date"]}', file=sf)
                print(f'', file=sf)

            episode = r.json().get('episode') or {}
            if episode:
                print('Episodes', file=sf)
                print(f'', file=sf)
                for key, item in episode.items():
                    print(f'#{item["episode"]} - {key!r} - {item["release_date"]}', file=sf)
                print(f'', file=sf)

        prompt = sf.getvalue()

    return {
        'page_id': page_id,
        'url': resp.url,
        'title': title,
        'prompt': prompt,
        'cover_image_url': cover_image_url,
        'synopsis': synopsis,
        'batch': list(batch.values()),
        'episode': list(episode.values()),
    }
