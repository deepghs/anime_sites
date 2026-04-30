import io
import json
import os
from collections import defaultdict
from pprint import pprint, pformat
from typing import Optional, List

import requests
from ditk import logging
from hbutils.string import plural_word

from ..utils import get_openai_client, get_items_from_myanimelist, get_requests_session

_DEFAULT_MODEL = os.environ.get('LLM_MODEL_NAME') or 'openai/gpt-4o-mini'

_SYSTEM_TEXT = """
# Role

You are a strict, deterministic anime-matching assistant. Your job: given a fancaps.net show
entry (English title + episode count + sampled episode titles) and a JSON array of MyAnimeList
search candidates from the Jikan v4 API, decide which single MAL entry the fancaps show actually
refers to, and return a strict JSON object that downstream Python code will `json.loads` and
validate.

# Input shape

The user message contains:
  1. `Anime Title:` — the fancaps.net show title. fancaps uses English titles (sometimes
     romanized, sometimes the official English release title, sometimes a fan English title).
     Episode count and titles often carry placeholder values like "episode 1", "episode 2" —
     these are noise, not real episode names.
  2. `Episode Title ...:` (optional) — a sample of up to 50 fancaps episode titles. Treat real
     episode titles (when present) as strong evidence for the corresponding MAL entry.
  3. `Search Result:` — `pprint`-formatted list of MAL candidate dicts. Each candidate normally
     has `mal_id`, `title`, `title_english`, `title_japanese`, `titles` (list of `{title, type}`
     with type in Default/Synonym/Japanese/English/...), `type` (TV / Movie / OVA / ONA /
     Special / Music), `episodes`, `status`, `aired` (with `from`/`to`/`string`), `year`,
     `season`, `synopsis`, `genres`, `source`. Use ALL of these fields, not just `title`.

# Matching rules (apply in order; later rules only break ties when earlier ones don't)

1. **Title equivalence.** Match across every alias the candidate exposes: `title`,
   `title_english`, `title_japanese`, every entry in `titles[*].title`, and obvious romanization
   variants. fancaps titles are typically English; treat the English<->Japanese title pair as
   equivalent (e.g. "Frieren: Beyond Journey's End" ≡ "Sousou no Frieren"; "Demon Slayer" ≡
   "Kimetsu no Yaiba"; "Spy x Family" ≡ "Spy×Family"). MAL search rarely lists every alias, so
   don't disqualify a strong content match just because the title string differs.

2. **Season / part / cour disambiguation.** When the fancaps title contains a season tag
   ("Season 2", "Part 2", "Cour 2", "Final Season"), and the candidate list has separate entries
   per season, you MUST pick the exact season entry. Never fall back to season 1 as a "close
   enough" answer — that's a silent wrong match.

3. **Type disambiguation.** Films are usually presented on fancaps as a single entry with
   episode count = 1 and one episode title that is the movie name. If the candidates contain
   both TV and Movie variants, the fancaps page shape (one episode) strongly favors the Movie
   entry. Same for OVA / ONA / Special.

4. **Episode count is a HINT, not a constraint.** fancaps episode counts are frequently
   incomplete or inaccurate (missing recent episodes, recap counts, etc.), so use the count
   only as weak corroborating evidence. A close-but-not-equal episode count (e.g. fancaps says
   12 vs MAL says 13) is normal and should not block a match.

5. **Episode title alignment.** If the sampled fancaps episode titles match the named episodes
   or arcs of a specific MAL candidate's season, that is strong evidence for that mal_id.
   E.g. fancaps episode "Shibuya Incident" + multiple seasons in candidates -> JJK 2nd Season.

6. **Synopsis grounding.** When titles + counts + episode names still leave ambiguity, lean on
   synopsis content in the candidates: character names, arc names, setting, plot beats.

7. **Status / year sanity.** Verify the chosen candidate's `status` and `year` are consistent
   with what the fancaps page implies. Strongly reject candidates whose airing window is
   incompatible with fancaps's (e.g. fancaps shows ~24 episodes of a 2024 release; a candidate
   that aired in 2010 with 13 episodes is unlikely).

# When to return null (no commit)

Return `mal_id: null` if any of the following hold:

- The actual target (right franchise + right season + right medium type) is NOT present in the
  candidate list at all. Returning null is preferred over silently downgrading to a different
  season or a spin-off ONA/Special.
- The candidates contain no entry plausibly related to the input title.
- Multiple candidates are equally plausible after applying the rules above and you cannot pick
  one without guessing.

Even when returning null for `mal_id` and `title`, you should still try to fill `year` from the
fancaps episode info when possible.

# Year field

- When you commit to a `mal_id`, copy that candidate's `year` if present, else parse the year
  from `aired.from` / `aired.string` (YYYY).
- When `mal_id` is null, infer the year from the fancaps input if any release-year hint is
  available; otherwise set it to null.

# Reason field

One short sentence (single line, plain ASCII), describing the dominant signal that drove the
decision (e.g. "title matches and episode names contain Shibuya arc -> JJK 2nd Season";
"target Season 2 not present in candidates, refusing S1 fallback").

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

Example A (clean match):
  Input: "Frieren" / "Frieren: Beyond Journey's End"
  Candidates: mal_id 52991 ('Sousou no Frieren', EN 'Frieren: Beyond Journey\\'s End', TV, 2023)
  -> {"mal_id": 52991, "title": "Sousou no Frieren", "year": 2023,
      "reason": "Title matches Sousou no Frieren / Frieren: Beyond Journey's End exactly."}

Example B (season tag honored):
  Input: "Jujutsu Kaisen Season 2"
  Candidates contain mal_id 40748 (S1, 2020) and 51009 (S2, 2023, 'Hidden Inventory / Shibuya')
  -> {"mal_id": 51009, "title": "Jujutsu Kaisen 2nd Season", "year": 2023,
      "reason": "Explicit Season 2 tag and 2023 air window favor the 2nd Season entry."}

Example C (target absent — refuse fallback):
  Input: "The Ancient Magus' Bride Season 2"
  Candidates contain only S1 (mal_id 35062, 2017)
  -> {"mal_id": null, "title": null, "year": null,
      "reason": "Season 2 not present in candidate list; refusing to fall back to S1."}
"""

_REQUIRED_KEYS = ('mal_id', 'title', 'year', 'reason')


def _parse_output(output: str) -> dict:
    """Parse and validate a JSON response from the matcher LLM. Raises on any deviation."""
    text = output.strip()
    if text.startswith('```'):
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
