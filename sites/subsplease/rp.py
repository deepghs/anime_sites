import datetime
import math
import os
import time
from typing import Optional, List, Tuple

import dateparser
import numpy as np
import pandas as pd
import requests
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import urlsplit, TemporaryDirectory
from hfutils.cache import delete_detached_cache
from hfutils.operate import get_hf_client, get_hf_fs, upload_directory_as_directory, download_directory_as_directory
from hfutils.utils import number_to_tag, hf_normpath
from huggingface_hub import hf_hub_url
from pyanimeinfo.myanimelist import JikanV4Client
from pyrate_limiter import Rate, Limiter, Duration

from .info import get_info_from_subsplease
from .match import _get_image_url
from ..utils import get_requests_session, parallel_call, download_file


def get_full_info_for_replace(page_url: str, mal_id: int, session: Optional[requests.Session] = None):
    session = session or get_requests_session()
    subs_info = get_info_from_subsplease(page_url, session=session)
    del subs_info['prompt']

    jikan_client = JikanV4Client(session=session)
    mal_info = jikan_client.get_anime_full(mal_id)

    min_timestamp = None
    for item in [*subs_info['batch'], *subs_info['episode']]:
        timestamp = dateparser.parse(item['release_date']).timestamp()
        if not min_timestamp or timestamp < min_timestamp:
            min_timestamp = timestamp
    year = datetime.datetime.fromtimestamp(min_timestamp).year if min_timestamp else None

    return {
        'mal_id': mal_info['mal_id'],
        'title': mal_info['title'],
        'reason': f'Admin specified {page_url!r} to mal #{mal_id}',
        'mal': mal_info,
        'year': year,
        'subsplease': subs_info,
    }


def sync(repository: str, change_items: List[Tuple[str, int]],
         upload_time_span: float = 30.0, deploy_span: float = 5 * 60.0,
         proxy_pool: Optional[str] = None, sync_mode: bool = True):
    delete_detached_cache()
    hf_client = get_hf_client()
    hf_fs = get_hf_fs()

    rate = Rate(1, int(math.ceil(Duration.SECOND * upload_time_span)))
    limiter = Limiter(rate, max_delay=1 << 32)

    if not hf_client.repo_exists(repo_id=repository, repo_type='dataset'):
        hf_client.create_repo(repo_id=repository, repo_type='dataset', private=True)
        attr_lines = hf_fs.read_text(f'datasets/{repository}/.gitattributes').splitlines(keepends=False)
        attr_lines.append('*.json filter=lfs diff=lfs merge=lfs -text')
        attr_lines.append('*.csv filter=lfs diff=lfs merge=lfs -text')
        hf_fs.write_text(
            f'datasets/{repository}/.gitattributes',
            os.linesep.join(attr_lines),
        )

    if hf_client.file_exists(
            repo_id=repository,
            repo_type='dataset',
            filename='table.parquet',
    ):
        df = pd.read_parquet(hf_client.hf_hub_download(
            repo_id=repository,
            repo_type='dataset',
            filename='table.parquet',
        )).replace(np.nan, None)
        d_animes = {item['page_id']: item for item in df.to_dict('records')}
    else:
        d_animes = {}

    session = get_requests_session()
    if proxy_pool:
        logging.info(f'Proxy pool {proxy_pool!r} enabled.')
        session.proxies.update({
            'http': proxy_pool,
            'https': proxy_pool
        })

    with TemporaryDirectory() as upload_dir:
        mal_covers_dir = os.path.join(upload_dir, 'assets', 'mal')
        os.makedirs(mal_covers_dir, exist_ok=True)
        subs_covers_dir = os.path.join(upload_dir, 'assets', 'subs')
        os.makedirs(subs_covers_dir, exist_ok=True)

        if sync_mode:
            logging.info('Downloading current mal images ...')
            download_directory_as_directory(
                repo_id=repository,
                repo_type='dataset',
                dir_in_repo=hf_normpath(os.path.relpath(mal_covers_dir, upload_dir)),
                local_directory=mal_covers_dir,
            )

            logging.info('Downloading current subsplease images ...')
            download_directory_as_directory(
                repo_id=repository,
                repo_type='dataset',
                dir_in_repo=hf_normpath(os.path.relpath(subs_covers_dir, upload_dir)),
                local_directory=subs_covers_dir,
            )

        _last_update, has_update = None, False
        _total_count = len(d_animes)

        def _deploy(force=False):
            nonlocal _last_update, has_update, _total_count

            if not has_update:
                return
            if not force and _last_update is not None and _last_update + deploy_span > time.time():
                return

            table_parquet_file = os.path.join(upload_dir, 'table.parquet')
            os.makedirs(os.path.dirname(table_parquet_file), exist_ok=True)
            df_animes = pd.DataFrame(list(d_animes.values())).replace(np.nan, None)
            df_animes['y'] = df_animes['year'].map(lambda x: x or 0)
            df_animes = df_animes.sort_values(by=['year', 'page_id'], ascending=[False, True])
            del df_animes['y']
            df_animes.to_parquet(table_parquet_file, engine='pyarrow', index=False)

            d_subs_images = {}

            def _fn_download_subs_cover(item):
                _, ext = os.path.splitext(urlsplit(item['subsplease_cover_image_url']).filename)
                dst_filename = os.path.join(subs_covers_dir, f'{item["page_id"]}{ext}')

                try:
                    if not os.path.exists(dst_filename):
                        download_file(
                            item['subsplease_cover_image_url'],
                            filename=dst_filename,
                            session=session,
                        )
                except:
                    if os.path.exists(dst_filename):
                        os.remove(dst_filename)
                    raise
                else:
                    d_subs_images[item['page_id']] = hf_normpath(os.path.relpath(dst_filename, upload_dir))

            parallel_call(
                df_animes[~df_animes['subsplease_cover_image_url'].isnull()].to_dict('records'),
                _fn_download_subs_cover,
                desc='Downloading Subsplease Cover Images'
            )

            d_mal_images = {}

            def _fn_download_mal_cover(item):
                _, ext = os.path.splitext(urlsplit(item['mal_cover_image_url']).filename)
                dst_filename = os.path.join(mal_covers_dir, f'{int(item["mal_id"])}{ext}')

                try:
                    if not os.path.exists(dst_filename):
                        download_file(
                            item['mal_cover_image_url'],
                            filename=dst_filename,
                            session=session,
                        )
                except:
                    if os.path.exists(dst_filename):
                        os.remove(dst_filename)
                    raise
                else:
                    d_mal_images[item['mal_id']] = hf_normpath(os.path.relpath(dst_filename, upload_dir))

            parallel_call(
                df_animes[~df_animes['mal_cover_image_url'].isnull()].to_dict('records'),
                _fn_download_mal_cover,
                desc='Downloading MAL Cover Images'
            )

            with open(os.path.join(upload_dir, 'README.md'), 'w') as f:
                print('---', file=f)
                print('license: other', file=f)
                print('language:', file=f)
                print('- en', file=f)
                print('- ja', file=f)
                print('tags:', file=f)
                print('- art', file=f)
                print('- anime', file=f)
                print('size_categories:', file=f)
                print(f'- {number_to_tag(len(df_animes))}', file=f)
                print('annotations_creators:', file=f)
                print('- no-annotation', file=f)
                print('source_datasets:', file=f)
                print('- subsplease', file=f)
                print('- myanimelist', file=f)
                print('---', file=f)
                print('', file=f)

                print(f'This is the matching result of subsplease and myanimelist, '
                      f'based on the LLM model.', file=f)
                print(f'', file=f)

                df_success = df_animes[~df_animes['mal_id'].isnull()].replace(np.nan, None)
                df_success = df_success.sort_values(by=['year', 'mal_id'], ascending=[False, False])
                if len(df_success):
                    print('## Resource', file=f)
                    print(f'', file=f)
                    print(f'{plural_word(len(df_success), "matched anime")} in total.', file=f)
                    print(f'', file=f)

                    lst_success = []
                    for item in df_success[:500].to_dict('records'):
                        if d_subs_images.get(item['page_id']):
                            subs_image_url = hf_hub_url(
                                repo_id=repository,
                                repo_type='dataset',
                                filename=d_subs_images[item['page_id']],
                            )
                        else:
                            subs_image_url = None
                        if d_mal_images.get(item['mal_id']):
                            mal_image_url = hf_hub_url(
                                repo_id=repository,
                                repo_type='dataset',
                                filename=d_mal_images[item['mal_id']],
                            )
                        else:
                            mal_image_url = None

                        subs_cover = f'![{item["page_id"]}]({subs_image_url})' if subs_image_url else 'N/A'
                        subs_title = item['subsplease_title']
                        if item['subsplease_url']:
                            subs_cover = f'[{subs_cover}]({item["subsplease_url"]})'
                            subs_title = f'[{subs_title}]({item["subsplease_url"]})'
                        mal_cover = f'![{item["mal_id"]}]({mal_image_url})' if mal_image_url else 'N/A'
                        mal_title = item['mal_title'] if item['mal_title'] else 'N/A'
                        if item.get('mal_url'):
                            mal_cover = f'[{mal_cover}]({item["mal_url"]})'
                            mal_title = f'[{mal_title}]({item["mal_url"]})'
                        lst_success.append({
                            'Subs Cover': subs_cover,
                            'Subs Title': subs_title,
                            'MAL ID': item['mal_id'],
                            'MAL Cover': mal_cover,
                            'MAL Title': mal_title,
                            'Year': int(item['year']) if item['year'] else 'N/A',
                            # 'Reason': item['reason'],
                        })

                    df_lst_success = pd.DataFrame(lst_success)
                    print(df_lst_success.to_markdown(index=False), file=f)
                    print(f'', file=f)

                df_failed = df_animes[df_animes['mal_id'].isnull()].replace(np.nan, None)
                if len(df_failed):
                    print('## Resources (Failed to Match)', file=f)
                    print(f'', file=f)
                    print(f'{plural_word(len(df_failed), "unmatched anime")} in total.', file=f)
                    print(f'', file=f)

                    lst_failed = []
                    for item in df_failed[:500].to_dict('records'):
                        if d_subs_images.get(item['page_id']):
                            subs_image_url = hf_hub_url(
                                repo_id=repository,
                                repo_type='dataset',
                                filename=d_subs_images[item['page_id']],
                            )
                        else:
                            subs_image_url = None

                        subs_cover = f'![{item["page_id"]}]({subs_image_url})' if subs_image_url else 'N/A'
                        subs_title = item['subsplease_title']
                        if item['subsplease_url']:
                            subs_cover = f'[{subs_cover}]({item["subsplease_url"]})'
                            subs_title = f'[{subs_title}]({item["subsplease_url"]})'
                        lst_failed.append({
                            'Subs Cover': subs_cover,
                            'Subs Title': subs_title,
                            'Year': int(item['year']) if item['year'] else 'N/A',
                            'Reason': item['reason'],
                        })

                    df_lst_failed = pd.DataFrame(lst_failed)
                    print(df_lst_failed.to_markdown(index=False), file=f)
                    print(f'', file=f)

            limiter.try_acquire('hf upload limit')
            upload_directory_as_directory(
                repo_id=repository,
                repo_type='dataset',
                local_directory=upload_dir,
                path_in_repo='.',
                message=f'Adding {plural_word(len(df_animes) - _total_count, "new record")} into index',
            )
            has_update = False
            _last_update = time.time()
            _total_count = len(df_animes)

        for sitem_url, mal_id in change_items:
            assert urlsplit(sitem_url).path_segments[1] == 'shows'
            page_id = urlsplit(sitem_url).path_segments[2]

            try:
                full_info = get_full_info_for_replace(sitem_url, mal_id, session=session)
            except:
                logging.exception(f'Error on {sitem_url!r}')
                continue
            row = {
                'page_id': page_id,
                'mal_id': full_info['mal_id'],
                'reason': full_info['reason'],
                'year': full_info['year'],
                **{f'subsplease_{key}': value for key, value in (full_info.get('subsplease') or {}).items()},
                **{f'mal_{key}': value for key, value in (full_info.get('mal') or {}).items()},
                'mal_cover_image_url': _get_image_url(full_info['mal']['images']) if full_info['mal'] else None,
            }
            d_animes[page_id] = row
            has_update = True

        _deploy(force=True)


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    sync(
        repository='deepghs/subsplease_mal',
        proxy_pool=os.environ['PP_SITE'],
        sync_mode=True,
        change_items=[
            # ('https://subsplease.org/shows/cardfight-vanguard-willdress/', 49819),
            # ('https://subsplease.org/shows/gunma-chan/', 49184),
            #
            # ('https://subsplease.org/shows/haikyuu-movie-gomisuteba-no-kessen/', 52742),
            # ('https://subsplease.org/shows/sengoku-youko/', 56242),
            # ('https://subsplease.org/shows/shy/', 53237),
            # ('https://subsplease.org/shows/mahoutsukai-no-yome-s2/', 52955),
            # ('https://subsplease.org/shows/gunma-chan/', 49184),
            # ('https://subsplease.org/shows/cardfight-vanguard-willdress/', 49819),
            # ('https://subsplease.org/shows/kyoukai-senki/', 48466),
            # ('https://subsplease.org/shows/shingeki-no-kyojin/', 40028),
            # ('https://subsplease.org/shows/idolish7-s3/', 45577),
            # ('https://subsplease.org/shows/mushoku-tensei/', 39535),
            # ('https://subsplease.org/shows/girls-und-panzer-das-finale/', 33970),
            #
            # ('https://subsplease.org/shows/another-journey-to-the-west/', 49757),
            # ('https://subsplease.org/shows/beheneko/', 58473),
            # ('https://subsplease.org/shows/boku-no-hero-academia-memories/', 57519),
            # ('https://subsplease.org/shows/dainanaoji/', 53516),
            # ('https://subsplease.org/shows/demon-slayer-kimetsu-no-yaiba/', 58473),
            # ('https://subsplease.org/shows/dragon-raja/', 44408),
            # ('https://subsplease.org/shows/high-card/', 49154),
            # # ('https://subsplease.org/shows/high-card/', 54869),
            # ('https://subsplease.org/shows/hime-sama-goumon-no-jikan-desu/', 55774),
            # ('https://subsplease.org/shows/link-click-bridon-arc/', 56752),
            # ('https://subsplease.org/shows/madome/', 53434),
            # ('https://subsplease.org/shows/megami-no-cafe-terrace/', 52973),
            # # ('https://subsplease.org/shows/megami-no-cafe-terrace/', 55749),
            # ('https://subsplease.org/shows/one-piece-fan-letter/', 60022),
            # ('https://subsplease.org/shows/tensei-shitara-slime-datta-ken/', 53580),
            # # ('https://subsplease.org/shows/tensei-shitara-slime-datta-ken/', 39551),
            # ('https://subsplease.org/shows/urusei-yatsura-2022/', 50710),
            # # ('https://subsplease.org/shows/urusei-yatsura-2022/', 54829),
            # ('https://subsplease.org/shows/dr-stone-s3/', 48549),
            # # ('https://subsplease.org/shows/dr-stone-s3/', 55644),
            # ('https://subsplease.org/shows/fate-strange-fake-whispers-of-dawn/', 53127),
            # ('https://subsplease.org/shows/heaven-officials-blessing-s2/', 50399),
            # ('https://subsplease.org/shows/ikenaikyo/', 52934),
            # ('https://subsplease.org/shows/jashin-chan-dropkick-seikimatsu-hen/', 55237),
            # ('https://subsplease.org/shows/kouryaku-wanted-isekai-sukuimasu/', 50571),
            # ('https://subsplease.org/shows/mashle/', 52211),
            # # ('https://subsplease.org/shows/kouryaku-wanted-isekai-sukuimasu/', 58473),
            # ('https://subsplease.org/shows/mobile-suit-gundam-cucuruz-doans-island/', 49827),
            # ('https://subsplease.org/shows/oshi-no-ko/', 52034),
            # # ('https://subsplease.org/shows/oshi-no-ko/', 55791),
            # ('https://subsplease.org/shows/x-and-y/', 50429),
            # ('https://subsplease.org/shows/dr-stone-ryuusui/', 50612),
            # ('https://subsplease.org/shows/girls-frontline/', 46604),
            # ('https://subsplease.org/shows/golden-kamuy/', 36028),
            # ('https://subsplease.org/shows/magia-record-final-season/', 40028),
            # ('https://subsplease.org/shows/mahouka-koukou-no-rettousei-tsuioku-hen/', 48375),
            # ('https://subsplease.org/shows/mobile-suit-gundam-the-witch-from-mercury/', 49828),
            # # ('https://subsplease.org/shows/mobile-suit-gundam-the-witch-from-mercury/', 53199),
            # ('https://subsplease.org/shows/shadowverse-flame/', 50060),
            # ('https://subsplease.org/shows/tokyo-mew-mew-new/', 41589),
            # # ('https://subsplease.org/shows/tokyo-mew-mew-new/', 53097),
            # ('https://subsplease.org/shows/bang-dream-movie-episode-of-roselia/', 41780),
            # # ('https://subsplease.org/shows/bang-dream-movie-episode-of-roselia/', 41781),
            # ('https://subsplease.org/shows/go-toubun-no-hanayome-s2/', 39783),
            # ('https://subsplease.org/shows/kaginado/', 48775),
            # # ('https://subsplease.org/shows/kaginado/', 50685),
            # ('https://subsplease.org/shows/muv-luv-alternative/', 40608),
            # # ('https://subsplease.org/shows/muv-luv-alternative/', 50638),
            # ('https://subsplease.org/shows/princess-principal-crown-handler/', 37807),
            # # ('https://subsplease.org/shows/thunderbolt-fantasy-s3/', NaN),
            # ('https://subsplease.org/shows/yakunara-mug-cup-mo/', 42568),
            # ('https://subsplease.org/shows/kings-raid-ishi-wo-tsugu-mono-tachi/', 41834),
            # ('https://subsplease.org/shows/tales-of-crestoria-toga-waga-wo-shoite-kare-wa-tatsu/', 42832),
            # ('https://subsplease.org/shows/yuukoku-no-moriarty/', 40911),
            # ('https://subsplease.org/shows/yuukoku-no-moriarty/', 43325),

            ('https://subsplease.org/shows/anne-shirley', 60334),
        ]
    )
