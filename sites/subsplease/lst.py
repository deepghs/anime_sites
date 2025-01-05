from typing import Optional, List
from urllib.parse import urljoin

import requests
from pyquery import PyQuery as pq

from ..utils import get_requests_session


def list_all_items_from_subsplease(session: Optional[requests.Session] = None) -> List[dict]:
    session = session or get_requests_session()
    resp = session.get('https://subsplease.org/shows/')
    page = pq(resp.text)

    retval = []
    for ei, show_item in list(enumerate(page('#main .all-shows .all-shows-link').items())):
        url = urljoin(resp.url, show_item('a').attr('href'))
        title = show_item('a').text().strip()
        retval.append({
            'title': title,
            'url': url,
        })

    return retval
