import io
import json
import os
from collections import defaultdict
from pprint import pformat
from typing import Optional, List

import requests
from ditk import logging
from hbutils.string import plural_word

from .info import get_info_from_subsplease
from ..utils import get_requests_session, get_openai_client, get_items_from_myanimelist

_DEFAULT_MODEL = os.environ.get('LLM_MODEL_NAME') or 'openai/gpt-4o-mini'

_SYSTEM_TEXT = """
# Role

You are a strict, deterministic anime-matching assistant. Your job: given a SubsPlease show entry
(title + synopsis + episode/release info) and a JSON array of MyAnimeList search candidates from
the Jikan v4 API, decide which single MAL entry the SubsPlease show actually refers to, and
return a strict JSON object that downstream Python code will `json.loads` and validate.

# Input shape

The user message contains:
  1. `Anime Title:` — the SubsPlease show title (English, romaji, or a mix; sometimes carries an
     explicit season tag like "S2", "2nd Season", "Part 2", "Final Season"; sometimes just the
     franchise base name even when SubsPlease is currently distributing a sequel/cour).
  2. `Anime Synopsis:` (optional) — synopsis text scraped from the SubsPlease page, often
     followed by a list of release rows like
       `#01 - 'Show Name - 01' - Tue, 05 Jul 2022 05:59:50 +0900`
     The release timestamps are extremely useful for season disambiguation: they tell you which
     years SubsPlease has actually been distributing this entry, which usually anchors which
     MAL season is the real target.
  3. `Search Result:` — `pprint`-formatted list of MAL candidate dicts. Each candidate normally
     has at minimum: `mal_id`, `title`, `title_english`, `title_japanese`, `titles` (list of
     `{title, type}` with type in Default/Synonym/Japanese/English/...), `type` (TV / Movie /
     OVA / ONA / Special / Music), `episodes`, `status`, `aired` (with `from`/`to`/`string`),
     `year`, `season`, `synopsis`, `genres`, `source`. Use ALL of these fields, not just `title`.

# Matching rules (apply in order; later rules only break ties when earlier ones don't)

1. **Title equivalence.** Match across every alias the candidate exposes: `title`,
   `title_english`, `title_japanese`, every entry in `titles[*].title`, and obvious romanization
   variants. Treat fan abbreviations as expanded (`JJK` ≡ `Jujutsu Kaisen`, `OreImo` ≡
   `Ore no Imouto ga Konnani Kawaii Wake ga Nai`, `SnK` ≡ `Shingeki no Kyojin`, `KonoSuba`,
   `Re:Zero`, `Mahoutsukai no Yome` ≡ `The Ancient Magus' Bride`, etc.). MAL search rarely lists
   every alias, so don't disqualify a strong content match just because the title string differs.

2. **Season / part / cour disambiguation.** When the SubsPlease title contains a season tag
   (`S2`, `S3`, `2nd Season`, `Part 2`, `Cour 2`, `Final Season`), and the candidate list has
   separate entries per season (e.g. `Foo`, `Foo 2nd Season`, `Foo Season 3`, `Foo Final Season
   Part 2`), you MUST pick the exact season entry. Never fall back to season 1 as a "close
   enough" answer — that's a silent wrong match.

3. **Type disambiguation.** If the synopsis or release pattern indicates a film (single release,
   "Movie", "Gekijouban") and the candidates contain both TV and Movie variants, pick the Movie
   entry. Same for OVA / ONA / Special. SubsPlease rarely distributes Music type; treat Music
   candidates as low-priority.

4. **SubsPlease season-tracking heuristic.** SubsPlease pages are *living* — the same page slug
   often gets reused across multiple seasons. When the SubsPlease title is just the franchise
   base name (no season tag), AND the release dates in the synopsis fall AFTER the airing
   window of the original/earliest MAL entry, the SubsPlease page is most likely tracking the
   *latest active* season, not season 1. In that case, pick the MAL entry whose `aired.from /
   aired.to` window contains the SubsPlease release timestamps. If the dates straddle multiple
   seasons (e.g. SubsPlease distributed both S1 and S2 over the years), prefer the *earliest*
   one whose airing window is consistent with the dates, unless the synopsis explicitly
   describes a later season.

5. **Synopsis grounding.** When titles + dates still leave ambiguity, lean on synopsis content:
   character names, arc names, setting, plot beats. A SubsPlease synopsis describing the
   "Pleiades Watchtower arc" matches the MAL candidate whose synopsis mentions the same arc.

6. **Status / year sanity check.** Verify that the chosen candidate's `status` and `year` are
   consistent with the SubsPlease release timestamps. Confidently reject candidates whose
   airing window is wholly before SubsPlease's release dates start (a finished show that ended
   in 2018 cannot be the target if SubsPlease's release rows are all 2024).

# When to return null (no commit)

Return `mal_id: null` if any of the following hold:

- The actual target (right franchise + right season + right medium type) is NOT present in the
  candidate list at all. Returning null is preferred over silently downgrading to a different
  season or a spin-off ONA/Special.
- The candidates contain no entry plausibly related to the input title.
- Multiple candidates are equally plausible after applying the rules above and you cannot pick
  one without guessing.

Even when returning null for `mal_id` and `title`, you should still try to fill `year` from the
SubsPlease release timestamps when possible.

# Year field

- When you commit to a `mal_id`, copy that candidate's `year` if present, else parse the year
  from `aired.from` / `aired.string` (YYYY).
- When `mal_id` is null, infer the year from the SubsPlease release timestamps in the synopsis
  (the year of the earliest episode release row).
- If literally no year can be inferred from anything, set `year: null`.

# Reason field

One short sentence (single line, plain ASCII), describing the dominant signal that drove the
decision (e.g. "title matches and 2024 release dates fall inside Season 3's airing window";
"title is generic but synopsis describes the Pleiades Watchtower arc which is unique to S3";
"target Season 2 not present in candidate list, only Season 1 available, refusing to fall back").

# Output format — STRICT JSON OBJECT

Return ONLY a single JSON object, no Markdown, no code fences, no commentary, no leading or
trailing whitespace beyond the JSON itself. The object MUST contain exactly these four keys:

```
{
  "mal_id": <integer or null>,
  "title":  <string or null>,
  "year":   <integer or null>,
  "reason": <one-line string>
}
```

Constraints:
- `mal_id` is either an integer that EXISTS in the candidate list's `mal_id` set, or `null`.
- `title` MUST be either the EXACT `title` string of the chosen candidate (copy it verbatim,
  including any non-ASCII characters), or `null` when `mal_id` is `null`.
- `year` is an integer (4 digits) or `null`.
- `reason` is a non-empty single-line string. Do not embed newlines.
- No additional keys, no nested objects, no arrays, no trailing commas.
- The output must round-trip through `json.loads(...)` with no errors.

# Worked examples

Example A (clean season match):
  Input title: "Re Zero S3"
  Synopsis: "...Pleiades Watchtower arc...", releases in 2024
  Candidates contain mal_id 31240 (S1, 2016), 39587 (S2, 2020), 56242 (S3, 2024)
  -> {"mal_id": 56242, "title": "Re:Zero kara Hajimeru Isekai Seikatsu 3rd Season",
      "year": 2024, "reason": "Title says S3 and synopsis + 2024 releases align with the
      Pleiades Watchtower / Season 3 entry."}

Example B (no season tag, latest cour):
  Input title: "Edens Zero" (no S2 tag)
  Releases dated 2023-08-... onward
  Candidates: mal_id 42192 (Edens Zero, 2021-2022), 50002 (Edens Zero 2nd Season, 2023)
  -> {"mal_id": 50002, "title": "Edens Zero 2nd Season", "year": 2023,
      "reason": "SubsPlease release dates (2023+) fall inside S2's airing window, so the page is
      tracking S2 even though the title omits the season tag."}

Example C (target absent — refuse to fall back):
  Input title: "Mahoutsukai no Yome Season 2"
  Candidates contain only the S1 entry (mal_id 35062, 2017)
  -> {"mal_id": null, "title": null, "year": 2023,
      "reason": "Season 2 not present in candidate list; refusing to fall back to S1."}
"""

_REQUIRED_KEYS = ('mal_id', 'title', 'year', 'reason')


def _parse_output(output: str) -> dict:
    """Parse and validate a JSON response from the matcher LLM. Raises on any deviation."""
    text = output.strip()
    if text.startswith('```'):
        # tolerate accidental markdown fence
        text = text.strip('`')
        if text.lstrip().startswith('json'):
            text = text.lstrip()[4:].lstrip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f'expected JSON object, got {type(data).__name__}')
    for k in _REQUIRED_KEYS:
        if k not in data:
            raise ValueError(f'missing key: {k!r}')
    if data['mal_id'] is not None and not isinstance(data['mal_id'], int):
        raise ValueError(f'mal_id must be int or null, got {data["mal_id"]!r}')
    if data['title'] is not None and not isinstance(data['title'], str):
        raise ValueError(f'title must be str or null, got {type(data["title"]).__name__}')
    if data['year'] is not None and not isinstance(data['year'], int):
        raise ValueError(f'year must be int or null, got {data["year"]!r}')
    if not isinstance(data['reason'], str) or not data['reason'].strip():
        raise ValueError(f'reason must be a non-empty string')
    return {k: data[k] for k in _REQUIRED_KEYS}


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
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {'role': 'system', 'content': _SYSTEM_TEXT},
                    {'role': 'user', 'content': message},
                ],
                response_format={'type': 'json_object'},
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
                 f'in {plural_word(val_times, "time")}: {dict(mal_ids)!r}'
        logging.warning(f'Match failed.\nReason: {reason}')
        return {
            'mal_id': None,
            'title': None,
            'reason': reason,
            'mal': None,
            'year': list(d_mal_vals.values())[0]['year'],
            'subsplease': subsplease_info,
        }
