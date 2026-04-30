import io
import json
import os
import re
from collections import defaultdict
from pprint import pprint, pformat
from typing import Optional, List

import requests
from ditk import logging
from hbutils.string import plural_word

from ..utils import get_openai_client, get_items_from_myanimelist, get_requests_session

_DEFAULT_MODEL = os.environ.get('LLM_MODEL_NAME') or 'openai/gpt-4o-mini'

_SYSTEM_TEXT = """
You are a strict anime-matching assistant. Given an anime title from fancaps.net (English-only,
sometimes with placeholder episode titles like "episode 1/2/3"), the number of fancaps episodes,
a sample of episode titles, and a JSON array of MyAnimeList search candidates, decide which single
MAL candidate corresponds to the input anime, and emit a structured result for an automated parser.

Matching rules (in priority order):
1. Title equivalence — match across all of: title, title_english, romaji/Japanese title, common
   abbreviations, and known alternative titles. fancaps titles are usually in English; treat
   English<->Japanese title pairs as equivalent.
2. Season / part / cour — if the fancaps title names a specific season ("Season 2", "Part 2",
   "Final Season", "Cour 2") and the candidates include separate entries per season, you MUST
   pick the exact season entry, not the franchise's first entry.
3. Movie vs TV — if fancaps describes a film (single feature) and the candidates contain both
   TV and Movie types, pick the Movie entry. Likewise for OVA / ONA / Special.
4. Episode count is a HINT, not a constraint — fancaps episode counts are frequently incomplete
   or inaccurate, so use them only as weak corroborating evidence.
5. Episode titles — if the sampled episode titles align with arcs/episode names of a specific
   MAL candidate's season, that is strong evidence for that mal_id.
6. Synopsis grounding (when present in candidates) — match story content, character names, arcs.

When NOT to commit (return null):
- The actual target anime / season / movie is NOT present in the search results, even if a
  closely-related entry (a different season of the same franchise, an unrelated movie, a spin-off)
  is present. DO NOT downgrade to "first season of the same franchise" as a fallback — that is a
  silent wrong match. Returning null is correct and preferred when the right entry is missing.
- The candidates contain no anime entry plausibly related to the input title at all.
- Multiple candidates are equally plausible and you cannot pick one with confidence.

Year field rules:
1. If you commit to a mal_id, prefer that candidate's 'year', else parse 'aired.from' (YYYY).
2. If you return null mal_id but can infer the input anime's release year from episode info,
   output that integer.
3. Only output `year: null` when no year can be inferred from anything.

Output format — exact, no other text, no Markdown, no code fences, no leading/trailing blank
lines, exactly four lines in this order:

mal_id: <integer or null>
title: <MAL title string, or null>
year: <integer or null>
reason: <one short single-line explanation; no line breaks>

Reply with ONLY those four lines. The downstream script parses them with strict regex; any
deviation (extra text, multi-line reason, missing field) is a hard failure.
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
