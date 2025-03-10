import os.path
from pprint import pprint
from typing import Optional
from urllib.parse import urljoin

import requests
from ditk import logging
from hbutils.system import urlsplit
from pyquery import PyQuery as pq

from ..utils import get_requests_session, srequest


def get_session():
    session = get_requests_session()
    session.headers.update({
        'Cookie': os.environ['ERAI_RAW_COOKIE'],
    })
    return session


def iter_anime_items(session: Optional[requests.Session] = None):
    session = session or get_session()
    resp = srequest(session, 'GET', 'https://www.erai-raws.info/anime-list/')
    page = pq(resp.text)

    count = 0
    for table in page('#main article .entry-content .tab-content div[id] table').items():
        for row in table('th a').items():
            anime_page_url = urljoin(resp.url, row.attr('href'))
            anime_title = row.text()
            yield anime_title, anime_page_url
            count += 1

    if count < 100:
        raise ValueError(f'Invalid list, should be no less than 100 but {count} found.')


def get_anime_info(anime_page_url: str, session: Optional[requests.Session] = None):
    session = session or get_session()
    resp = srequest(session, 'GET', anime_page_url)
    page = pq(resp.text)

    main = page('#main')
    content = main('.entry-content')
    title = main('h1.entry-title').text().strip()
    poster_url = urljoin(resp.url, content('.entry-content-poster img').attr('src'))
    story = content('.entry-content-story').text()

    external_links = {}
    for btn_group in content('.entry-content-buttons').items():
        if 'more info:' in btn_group.text().lower():
            for btn_item in btn_group('.entry-sub-content-buttons').items():
                external_links[btn_item.text().strip()] = urljoin(resp.url, btn_item.attr('href'))

    if 'MAL' in external_links:
        segments = urlsplit(external_links['MAL']).path_segments
        assert segments[1] == 'anime', f'Anime expected but {external_links["MAL"]!r} found.'
        mal_id = int(segments[2])
    else:
        mal_id = None

    return {
        'mal_id': mal_id,
        'title': title,
        'poster_url': poster_url,
        'story': story,
        'external_links': external_links,
    }


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    session = get_session()
    for item in iter_anime_items(session=session):
        print(item)
        pprint(get_anime_info(item[1]))
        quit()
