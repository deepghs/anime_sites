import math
import os
import re
import time
from typing import Optional

import numpy as np
import pandas as pd
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import urlsplit, TemporaryDirectory
from hfutils.cache import delete_detached_cache
from hfutils.operate import get_hf_client, get_hf_fs, upload_directory_as_directory, download_directory_as_directory
from hfutils.utils import number_to_tag, hf_normpath
from huggingface_hub import hf_hub_url
from pyrate_limiter import Rate, Limiter, Duration

from .llm import get_full_info_for_subsplease
from .lst import list_all_items_from_subsplease
from ..utils import get_requests_session, parallel_call, download_file


def _get_url_from_small_dict(dict_: dict):
    if 'maximum_image_url' in dict_:
        return dict_['maximum_image_url']
    elif 'large_image_url' in dict_:
        return dict_['large_image_url']
    elif 'small_image_url' in dict_:
        return dict_['small_image_url']
    else:
        return dict_['image_url']


def _get_image_url(image_dict: dict):
    if 'jpg' in image_dict:
        return _get_url_from_small_dict(image_dict['jpg'])
    elif 'webp' in image_dict:
        return _get_url_from_small_dict(image_dict['webp'])
    else:
        return None


def _name_safe(name_text):
    return re.sub(r'[\W_]+', ' ', name_text).strip(' ')


def sync(repository: str, upload_time_span: float = 30.0, deploy_span: float = 5 * 60.0,
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
                    for item in df_success.to_dict('records'):
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
                            'Reason': item['reason'],
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
                    for item in df_failed.to_dict('records'):
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

        for sitem in list_all_items_from_subsplease(session=session):
            assert urlsplit(sitem['url']).path_segments[1] == 'shows'
            page_id = urlsplit(sitem['url']).path_segments[2]

            if page_id in d_animes and d_animes[page_id]['mal_id']:
                logging.warning(f'Anime {sitem!r} already matched, skipped.')
                continue
            elif not sync_mode and page_id in d_animes:
                logging.warning(f'Anime {sitem!r} already asked, but not matched, skipped due to non-sync mode.')
                continue

            full_info = get_full_info_for_subsplease(sitem['url'], session=session)
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
            _deploy()

        _deploy(force=True)


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    sync(
        repository='deepghs/subsplease_mal',
        deploy_span=5 * 60.0,
        proxy_pool=os.environ['PP_SITE'],
        sync_mode=False,
    )
