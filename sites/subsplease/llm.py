import io
import json
import re
import time
import warnings
from collections import defaultdict
from pprint import pformat
from typing import Optional, List

import requests
from ditk import logging
from hbutils.string import plural_word
from pyanimeinfo.myanimelist import JikanV4Client
from requests import HTTPError

from sites.subsplease.info import get_info_from_subsplease
from sites.utils import get_requests_session, get_openai_client

_DEFAULT_MODEL = 'openai/gpt-4o'


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


_SYSTEM_TEXT = """
Translation of the provided content for use in a large language model prompt: You are an anime information 
filtering assistant. Based on the anime titles I provide (which may be abbreviated titles), 
the synopsis of the anime, and the results I retrieve from the MyAnimeList website's search API (in JSON format), 
determine which one of the search results corresponds to the anime I am looking for, and output the corresponding 
'mal_id' and 'title' and 'year' values.

If there are multiple potential matches, please identify the best match based on the information provided. 
If none of them can be matched, then return "No match found".

Translation: The key aspects you should check include:
1. Title, to see if any names, aliases, or names in other languages provided in the search results match.
   (but sometimes the title will have many different aliases which the search result not included, please attention that)
2. Synopsis (if provided), to check if the main content and summary of the anime provided are essentially consistent with those provided.
3. Status (if episode information is provided), to verify whether the current status of the anime 
   (Not yet aired/Currently Airing/Finished Airing) matches with the search results.
4. Year (if episode information is provided), to check if the year information of each episode of the anime 
   matches with the years provided in the search results.

For the year field:
1. It should be an integer, mainly inferred from the anime information provided.
2. If you can find this information in the matched search result, use the year from the search result.
3. If no specific year is provided in the matched search result, try find it from the 'aired' item from the matched search result.
   Use the 'aired.from' time as the result.
4. If you cannot find this information in any parts of the search result, or no matching search result can be found,
   Just use the information you inferred from the provided information of the anime episodes.
   The year result should be the year which the first episode is released.
5. If there is truly no information at all, return the "null". 

When best match is found, reply as the following format:

mal_id: xxxxx (should be an integer)
title: xxxxxxxxxx (should be a string)
year: xxxx (should be an integer, mainly inferred from the anime information provided)
reason: xxxxxx

The Reason should be in a line.

When no match is found, reply as the following format:

mal_id: null
title: null
year: xxxx/null (no matter match success or not, you should try your best to infer the year of this anime, unless truly impossible)
reason: xxxxxxxxx

DO NOT OUTPUT ANYTHING ELSE EXCEPT THESE, MAKE SURE THE OUTPUT CAN BE PROCESSED BY THE AUTOMATED SCRIPT.
NO MATTER YOU FIND THE MATCH OR NOT, PLEASE DESCRIBE YOU REASONS AND WHY YOU GIVE THIS ANSWER. 
"""

_NOT_SET = object()


def _parse_output(output: str):
    mal_id, title, year, reason = _NOT_SET, _NOT_SET, _NOT_SET, _NOT_SET
    for line in output.strip().splitlines(keepends=False):
        line = line.strip()
        if mal_id is _NOT_SET:
            if line:
                matching = re.fullmatch(r'^mal_id\s*:\s*(?P<id>\d+|null)$', line)
                mal_id = json.loads(matching.group('id'))
        elif title is _NOT_SET:
            if line:
                matching = re.fullmatch(r'^title\s*:\s*(?P<title>[\s\S]+?)\s*$', line)
                title = matching.group('title')
        elif year is _NOT_SET:
            if line:
                matching = re.fullmatch(r'^year\s*:\s*(?P<year>\d+|null)$', line)
                year = json.loads(matching.group('year'))
        else:
            if reason is _NOT_SET:
                matching = re.fullmatch(r'^reason\s*:\s*(?P<reason>[\s\S]*?)\s*$', line)
                reason = matching.group('reason')
            else:
                reason += '\n' + line

    assert mal_id is not _NOT_SET, 'mal_id not found'
    assert title is not _NOT_SET, 'title not found'
    assert year is not _NOT_SET, 'year not found'
    assert reason is not _NOT_SET, 'reason not found'
    return {
        'mal_id': mal_id,
        'title': title,
        'year': year,
        'reason': reason,
    }


def _ask_chatgpt(title: str, synopsis: Optional[str] = None, search_result: Optional[List[dict]] = None,
                 model_name: str = _DEFAULT_MODEL, max_tries: int = 5):
    client = get_openai_client()

    if search_result is None:
        search_result = get_items_from_myanimelist(title)
    d_items = {item['mal_id']: item for item in search_result}

    with io.StringIO() as sf:
        print(f'Anime Title: {title!r}', file=sf)
        print(f'', file=sf)
        if synopsis:
            print(f'Anime Synopsis:', file=sf)
            print(f'{synopsis}', file=sf)
            print(f'', file=sf)

        print(f'Search Result:', file=sf)
        print(pformat(search_result), file=sf)
        print(f'', file=sf)

        message = sf.getvalue()

    tries = 0
    while tries < max_tries:
        logging.info(f'Asking LLM model {model_name!r} about {title!r} ...')
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {'role': 'system', 'content': _SYSTEM_TEXT},
                {"role": "user", "content": message},
            ],
        )
        resp_text = response.choices[0].message.content.strip()
        logging.info(f'Response from LLM:\n{resp_text}')

        try:
            pinfo = _parse_output(resp_text)
            if pinfo['mal_id'] and pinfo['mal_id'] in d_items:
                return {
                    **pinfo,
                    'year': d_items[pinfo['mal_id']]['year'] or pinfo['year'],
                    'mal': d_items[pinfo['mal_id']],
                }
            else:
                return {
                    'mal_id': None,
                    'title': None,
                    'reason': pinfo['reason'],
                    'year': pinfo['year'],
                    'mal': None,
                }
        except Exception as err:
            tries += 1
            logging.error(f'({tries}/{max_tries}) Error when parsing output - {err!r}')
            continue

    raise RuntimeError(f'Unable to get result for {title!r}')


def get_full_info_for_subsplease(url, model_name: str = _DEFAULT_MODEL, val_times: int = 5, min_val: int = 4,
                                 session: Optional[requests.Session] = None):
    session = session or get_requests_session()
    info = get_info_from_subsplease(url, session=session)
    search_result = get_items_from_myanimelist(info['title'], session=session)
    subsplease_info = {
        'url': url,
        **{key: value for key, value in info.items() if key != 'prompt'},
    }

    vals = []
    mal_ids = defaultdict(lambda: 0)
    d_mal_vals = {}
    for i in range(val_times):
        logging.info(f'Val {i + 1} / {val_times} for {info["title"]!r} ...')
        val = _ask_chatgpt(info['title'], synopsis=info['prompt'],
                           search_result=search_result, model_name=model_name)
        vals.append(val)
        mal_ids[val['mal_id']] += 1
        if val['mal_id'] not in d_mal_vals:
            d_mal_vals[val['mal_id']] = val

    if None in mal_ids:
        del mal_ids[None]

    for mal_id, count in mal_ids.items():
        if mal_id and count >= min_val:
            logging.info(f'Match success, the final result is #{mal_id!r}.\n'
                         f'Reason: {d_mal_vals[mal_id]["reason"]}')
            return {
                **d_mal_vals[mal_id],
                'subsplease': subsplease_info
            }

    if None in d_mal_vals:
        reason = d_mal_vals[None]["reason"]
        logging.warning(f'Match failed.\nReason: {reason}')
        return {
            **d_mal_vals[None],
            'subsplease': subsplease_info,
        }
    else:
        reason = f'Cannot determine which anime it is due to the complex result ' \
                 f'in {plural_word(val_times, "time")}: {mal_ids!r}'
        logging.warning(f'Match failed.\nReason: {reason}')
        return {
            'mal_id': None,
            'title': None,
            'reason': reason,
            'mal': None,
            'year': list(d_mal_vals.values())[0]['year'],
            'subsplease': subsplease_info,
        }
