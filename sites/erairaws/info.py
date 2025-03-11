import json
import os.path
from pprint import pprint
from typing import Optional, Union, List
from urllib.parse import urljoin

import dateparser
import requests
from ditk import logging
from hbutils.system import urlsplit
from pyquery import PyQuery as pq

from ..utils import get_requests_session, srequest


def get_session(no_login: bool = False):
    session = get_requests_session()
    if not no_login:
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


def get_anime_info(anime_page_url: str, session: Optional[requests.Session] = None,
                   session_rss: Optional[Union[List[requests.Session], requests.Session]] = None):
    session = session or get_session(no_login=False)
    session_rss = session_rss or get_session(no_login=True)
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
    other_links = {}
    rss_url = None
    for btn_group in content('.entry-content-buttons').items():
        if 'more info:' in btn_group.text().lower():
            for btn_item in btn_group('.entry-sub-content-buttons').items():
                external_links[btn_item.text().strip()] = urljoin(resp.url, btn_item.attr('href'))
        elif 'rss link' in btn_group.text().lower():
            for pitem in btn_group('p').items():
                if 'rss link' in pitem.text().lower() and 'ALL' in pitem.text():
                    rss_url = urljoin(resp.url, pitem('a').attr('href'))
        elif 'other:' in btn_group.text().lower():
            for btn_item in btn_group('a.entry-sub-content-buttons').items():
                other_links[btn_item.text().strip()] = urljoin(resp.url, btn_item.attr('href'))

    if 'MAL' in external_links:
        segments = urlsplit(external_links['MAL']).path_segments
        assert segments[1] == 'anime', f'Anime expected but {external_links["MAL"]!r} found.'
        mal_id = int(segments[2])
    else:
        mal_id = None

    r_pane = None
    for pane in main('.tab-content > .tab-pane').items():
        if 'all release' in pane('h4.alphabet-title').text().strip().lower():
            r_pane = pane
            break

    resources = []
    for table in r_pane('table.table').items():
        categories = [
            c.attr('data-title')
            for c in table('tr:nth-child(1) th a[data-title]').items()
        ]

        ititle = table('tr:nth-child(1) th a.aa_ss_ops_new').text().strip()
        iurl = urljoin(resp.url, table('tr:nth-child(1) th a.aa_ss_ops_new').attr('href'))

        sec_links = {
            x.text().strip(): urljoin(resp.url, x('a').attr('href')) if 'magnet' not in x.text().strip().lower() else x(
                'a').attr('href')
            for x in table('tr:nth-child(2) th a.sub_ddl_box').items()
        }
        langs = [
            x.attr('data-title')
            for x in table('tr:nth-child(2) th span.tooltip3[data-title]').items()
        ]
        publish_at_str = table('tr:nth-child(3) th font.clock_font').text()
        published_at = dateparser.parse(publish_at_str).timestamp()

        rurls = {}
        rx_maps = {}
        for sitem in table('tr:nth-child(3) th span').items():
            if sitem('a').attr('href'):
                rurls[sitem('a').text().strip()] = \
                    urljoin(resp.url, sitem('a').attr('href')) if 'magnet' not in sitem('a').text().lower() else sitem(
                        'a').attr('href')
            else:
                rx = sitem('a').text()
                rx_maps[rx] = sitem.attr('id')

        if rx_maps:
            for rx, rx_id in rx_maps.items():
                span_text = table(f'tr[class~={json.dumps(rx_id)}] th > span:nth-child(1)').text()
                span_segs = span_text.split('|', maxsplit=2)
                size_text = None
                ext_info = None
                for seg in span_segs:
                    seg = seg.strip()
                    if 'size' in seg.lower():
                        size_text = seg.split(':', maxsplit=1)[-1].strip()
                    elif size_text is not None:
                        ext_info = seg

                rurls[rx] = {
                    'size': size_text,
                    'ext': ext_info,
                }
                for ax in table(f'tr[class~={json.dumps(rx_id)}] th > a').items():
                    rurls[rx][ax.text().strip()] = \
                        urljoin(resp.url, ax.attr('href')) if 'magnet' not in ax.text().lower() else \
                            ax.attr('href')

        item = {
            'title': ititle,
            'page_url': iurl,
            'categories': categories,
            'sec_links': sec_links,
            'langs': langs,
            'published_at': published_at,
            'resource_urls': rurls,
        }
        resources.append(item)

    return {
        'id': id_,
        'page_url': resp.url,
        'mal_id': mal_id,
        'title': title,
        'poster_url': poster_url,
        'story': story,
        'external_links': external_links,
        'other_links': other_links,
        'related': related,
        'rss_url': rss_url,
        'resources': resources,
        'last_published_at': max(x['published_at'] for x in resources) if resources else None,
        'published_at': min(x['published_at'] for x in resources) if resources else None,
    }


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    session = get_session()
    pprint(get_anime_info('https://www.erai-raws.info/anime-list/rezero-kara-hajimeru-isekai-seikatsu-3rd-season/'))
    # for item in iter_anime_items(session=session):
    #     print(item)
    #     pprint(get_anime_info(item[1]))
    #     quit()
