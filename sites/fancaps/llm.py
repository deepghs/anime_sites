import io
import json
import re
from collections import defaultdict
from pprint import pprint, pformat
from typing import Optional, List

import requests
from ditk import logging
from hbutils.string import plural_word

from ..utils import get_openai_client, get_items_from_myanimelist, get_requests_session

_DEFAULT_MODEL = 'openai/gpt-4o'

_SYSTEM_TEXT = """
You are an effective assistant for matching anime information between fancaps.net and myanimelist websites. 
I will provide an anime id/title (in English) from fancaps.net, along with the number of episodes included on 
the fancaps website and some episode titles (representing only fancaps data, episodes may be missing; also, 
some anime titles might have meaningless content like "episode 1/2/3"), and search results from myanimelist 
website (in JSON format). You need to identify the most accurate match for the fancaps anime from the search results. 
including its mal_id, title from mal and year.

Please note the following:
1. Prioritize matching based on the anime's name, translations, and alternative titles
2. You can reference episode count information, but keep in mind that data on fancaps website may not be 
   absolutely accurate, so this point cannot be definitive
3. You can also check the synopsis (if provided), to check if the main content and summary of the anime provided 
   are essentially consistent with those provided.
3. If there are multiple potential matches, please provide only one most accurate match, such as a specific 
   season within the same anime series
4. There may be cases where no matching result can be found, in which case you should output "none" as the result, 
   rather than forcing a match with an unrelated result

For the year field:
1. It should be an integer, mainly inferred from the anime information provided.
2. If you can find this information in the matched search result, use the year from the search result.
3. If no specific year is provided in the matched search result, try find it from the 'aired' item from the matched search result.
   Use the 'aired.from' time as the result.
4. If you cannot find this information in any parts of the search result, or no matching search result can be found,
   Just use the information you inferred from the provided information of the anime episodes.
   The year result should be the year which the first episode is released.
5. If there is truly no information at all, return the "null". 

When the best match is found, output in the following format:

mal_id: xxxxx (should be an integer)
title: xxxxxxxxxx (should be a string)
year: xxxx (should be an integer, mainly inferred from the anime information provided)
reason: xxxxxx (The reason should be strictly on a single line, multiple lines for reasons are not allowed)

When no search result is found, output in the following format:

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


def _ask_chatgpt(bg_item, search_result: Optional[List[dict]] = None,
                 model_name: str = _DEFAULT_MODEL, max_tries: int = 5):
    client = get_openai_client()
    title = bg_item['title']
    episode_titles = [x['title'] for x in bg_item['episodes']]

    if search_result is None:
        search_result = get_items_from_myanimelist(title)
    d_items = {item['mal_id']: item for item in search_result}

    with io.StringIO() as sf:
        print(f'Anime Title: {title!r}', file=sf)
        print(f'', file=sf)
        if episode_titles:
            print(f'Episode Title ({plural_word(len(episode_titles), "episode")} in total, '
                  f'only first {len(episode_titles[:50])} are shown):', file=sf)
            for et in episode_titles[:50]:
                print(f'- {et!r}', file=sf)
            print(f'', file=sf)

        print(f'Search Result:', file=sf)
        print(pformat(search_result), file=sf)
        print(f'', file=sf)

        message = sf.getvalue()

    tries = 0
    while tries < max_tries:
        logging.info(f'Asking LLM model {model_name!r} about {title!r} ...')
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {'role': 'system', 'content': _SYSTEM_TEXT},
                    {"role": "user", "content": message},
                ],
            )
            resp_text = response.choices[0].message.content.strip()
            logging.info(f'Response from LLM:\n{resp_text}')

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


def get_full_info_for_fancaps(bg_item, model_name: str = _DEFAULT_MODEL, val_times: int = 5, min_val: int = 4,
                              session: Optional[requests.Session] = None):
    session = session or get_requests_session()
    search_result = get_items_from_myanimelist(bg_item['title'], session=session)

    vals = []
    mal_ids = defaultdict(lambda: 0)
    d_mal_vals = {}
    for i in range(val_times):
        logging.info(f'Val {i + 1} / {val_times} for {bg_item["title"]!r} ...')
        val = _ask_chatgpt(bg_item, search_result=search_result, model_name=model_name)
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
                'fancaps': bg_item
            }

    if None in d_mal_vals:
        reason = d_mal_vals[None]["reason"]
        logging.warning(f'Match failed.\nReason: {reason}')
        return {
            **d_mal_vals[None],
            'fancaps': bg_item,
        }
    else:
        reason = f'Cannot determine which anime it is due to the complex result ' \
                 f'in {plural_word(val_times, "time")}: {dict(mal_ids)!r}'
        logging.warning(f'Match failed.\nReason: {reason}')
        return {
            'mal_id': None,
            'title': None,
            'reason': reason,
            'mal': None,
            'year': list(d_mal_vals.values())[0]['year'],
            'fancaps': bg_item,
        }


if __name__ == '__main__':
    logging.try_init_root(level=logging.INFO)
    from .data import _get_mappings

    bgs = _get_mappings()
    pprint(get_full_info_for_fancaps(bg_item=bgs[-1]))
    # pprint(get_items_from_myanimelist('The Girl in Twilight'))
