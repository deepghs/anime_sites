import math
import os
import re
import time

import pandas as pd
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import urlsplit, TemporaryDirectory
from hfutils.cache import delete_detached_cache
from hfutils.operate import get_hf_client, get_hf_fs, upload_directory_as_directory
from hfutils.utils import number_to_tag
from pyrate_limiter import Rate, Limiter, Duration

from .llm import get_full_info_for_subsplease
from .lst import list_all_items_from_subsplease
from ..utils import get_requests_session


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


def sync(repository: str, upload_time_span: float = 30.0, deploy_span: float = 5 * 60.0):
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
        ))
        d_animes = {item['page_id']: item for item in df.to_dict('records')}
    else:
        d_animes = {}

    session = get_requests_session()
    with TemporaryDirectory() as upload_dir:
        mal_covers_dir = os.path.join(upload_dir, 'assets', 'mal')
        os.makedirs(mal_covers_dir, exist_ok=True)
        subs_covers_dir = os.path.join(upload_dir, 'assets', 'subs')
        os.makedirs(subs_covers_dir, exist_ok=True)

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
            df_animes = pd.DataFrame(list(d_animes.values()))
            df_animes = df_animes.sort_values(by=['page_id'], ascending=[False])
            df_animes.to_parquet(table_parquet_file, engine='pyarrow', index=False)


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

                # print('## Records', file=f)
                # print(f'', file=f)
                # df_records_shown = df_animes[:50][
                #     ['id', 'image_width', 'image_height', 'rating', 'mimetype', 'file_size', 'file_url']]
                # print(f'{plural_word(len(d_records), "record")} in total. '
                #       f'Only {plural_word(len(df_records_shown), "record")} shown.', file=f)
                # print(f'', file=f)
                # print(df_records_shown.to_markdown(index=False), file=f)
                # print(f'', file=f)

            limiter.try_acquire('hf upload limit')
            # upload_directory_as_directory(
            #     repo_id=repository,
            #     repo_type='dataset',
            #     local_directory=upload_dir,
            #     path_in_repo='.',
            #     hf_token=os.environ['HF_TOKEN'],
            #     message=f'Adding {plural_word(len(df_animes) - _total_count, "new record")} into index',
            # )
            has_update = False
            _last_update = time.time()
            _total_count = len(df_animes)

        for sitem in list_all_items_from_subsplease(session=session):
            assert urlsplit(sitem['url']).path_segments[1] == 'shows'
            page_id = urlsplit(sitem['url']).path_segments[2]

            if page_id in d_animes and d_animes[page_id]['mal_id']:
                logging.warning(f'Anime {sitem!r} already matched, skipped.')
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

            if len(d_animes) >= 2:
                break

        _deploy()


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    sync(
        repository='deepghs/subsplease_mal',
    )
