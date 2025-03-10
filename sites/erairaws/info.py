import os.path
from pprint import pprint
from typing import Optional
from urllib.parse import urljoin

import requests
import xmltodict
from ditk import logging
from hbutils.system import urlsplit
from pyquery import PyQuery as pq
from urlobject import URLObject

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

    psplit = urlsplit(resp.url)
    assert psplit.path_segments[1] == 'anime-list'
    id_ = psplit.path_segments[2]

    main = page('#main')
    content = main('.entry-content')
    title = main('h1.entry-title').text().strip()
    poster_url = urljoin(resp.url, content('.entry-content-poster img').attr('src'))
    story = content('.entry-content-story').text()

    related = []
    for related_item in content('.entry-content-related ul li').items():
        related.append({
            'title': related_item('a').text().strip(),
            'url': urljoin(resp.url, related_item('a').attr('href')),
        })

    external_links = {}
    rss_url = None
    for btn_group in content('.entry-content-buttons').items():
        if 'more info:' in btn_group.text().lower():
            for btn_item in btn_group('.entry-sub-content-buttons').items():
                external_links[btn_item.text().strip()] = urljoin(resp.url, btn_item.attr('href'))
        elif 'rss link' in btn_group.text().lower():
            for pitem in btn_group('p').items():
                if 'rss link' in pitem.text().lower() and 'ALL' in pitem.text():
                    rss_url = urljoin(resp.url, pitem('a').attr('href'))

    if 'MAL' in external_links:
        segments = urlsplit(external_links['MAL']).path_segments
        assert segments[1] == 'anime', f'Anime expected but {external_links["MAL"]!r} found.'
        mal_id = int(segments[2])
    else:
        mal_id = None

    ress = ['1080p', '720p', 'SD']
    magnet_url = URLObject(rss_url).add_query_param('type', 'magnet')
    rss_items_res = None
    rss_items = []
    for res in ress:
        rss_item_url = str(magnet_url.add_query_param('res', res))
        r = session.get(rss_item_url)
        r.raise_for_status()

        rd = xmltodict.parse(r.text)
        items = rd['rss']['channel'].get('item')
        if items:
            if not isinstance(items, list):
                items = [items]
            rss_items = items
            rss_items_res = res
            break

    return {
        'id': id_,
        'page_url': resp.url,
        'mal_id': mal_id,
        'title': title,
        'poster_url': poster_url,
        'story': story,
        'external_links': external_links,
        'related': related,
        'rss_url': rss_url,
        'rss_items_res': rss_items_res,
        'rss_items': rss_items,
    }


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    session = get_session()
    pprint(get_anime_info('https://www.erai-raws.info/anime-list/rezero-kara-hajimeru-isekai-seikatsu-3rd-season/'))
    # for item in iter_anime_items(session=session):
    #     print(item)
    #     pprint(get_anime_info(item[1]))
    #     quit()
