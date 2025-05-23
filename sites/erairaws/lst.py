import datetime
import os.path
from threading import Lock
from typing import Optional
from urllib.parse import unquote_plus, quote

import numpy as np
import pandas as pd
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory, urlsplit
from hfutils.cache import delete_detached_cache
from hfutils.operate import download_directory_as_directory, get_hf_client, get_hf_fs, upload_directory_as_directory
from hfutils.utils import number_to_tag
from huggingface_hub import hf_hub_url

from .info import get_session, iter_anime_items, get_anime_info
from ..utils import parallel_call, download_file


def _url_safe(url):
    return quote(unquote_plus(url), safe=':/?#[]@!$&\'()*+,;=').replace('(', '%28').replace(')', '%29')


def sync(repository: str, proxy_pool: Optional[str] = None):
    delete_detached_cache()
    hf_client = get_hf_client()
    hf_fs = get_hf_fs()

    if not hf_client.repo_exists(repo_id=repository, repo_type='dataset'):
        hf_client.create_repo(repo_id=repository, repo_type='dataset', private=True)
        attr_lines = hf_fs.read_text(f'datasets/{repository}/.gitattributes').splitlines(keepends=False)
        attr_lines.append('*.json filter=lfs diff=lfs merge=lfs -text')
        attr_lines.append('*.csv filter=lfs diff=lfs merge=lfs -text')
        hf_fs.write_text(
            f'datasets/{repository}/.gitattributes',
            os.linesep.join(attr_lines),
        )

    session = get_session(no_login=False)
    session_rss = [get_session(no_login=True) for _ in range(100)]
    if proxy_pool:
        logging.info(f'Proxy pool {proxy_pool!r} enabled.')
        session.proxies.update({
            'http': proxy_pool,
            'https': proxy_pool
        })
        for ss in session_rss:
            ss.proxies.update({
                'http': proxy_pool,
                'https': proxy_pool
            })

    anime_items = list(iter_anime_items(session=session))
    anime_records = []
    item_records = []
    lock = Lock()

    def _fn_x(ax):
        title, page_url = ax
        logging.info(f'Processing {title!r}, page: {page_url!r} ...')
        info = get_anime_info(page_url, session=session, session_rss=session_rss)
        if not info['mal_id']:
            logging.warning(f'No MAL ID found for {page_url!r}, skipped.')
            return

        with lock:
            _, ext = os.path.splitext(urlsplit(info['poster_url']).filename)
            anime_records.append({
                **info,
                'poster_filename': f'{info["id"]}{ext}',
            })
            for item in info['resources']:
                item_records.append({
                    'mal_id': info['mal_id'],
                    **item,
                })

    parallel_call(anime_items, _fn_x, desc='Animes')

    # for title, page_url in tqdm(anime_items, desc='Animes'):
    #     logging.info(f'Processing {title!r}, page: {page_url!r} ...')
    #
    #     info = get_anime_info(page_url, session=session, session_rss=session_rss)
    #     if not info['mal_id']:
    #         logging.warning(f'No MAL ID found for {page_url!r}, skipped.')
    #         continue
    #
    #     _, ext = os.path.splitext(urlsplit(info['poster_url']).filename)
    #     anime_records.append({
    #         **info,
    #         'poster_filename': f'{info["id"]}{ext}',
    #     })
    #     for item in info['resources']:
    #         item_records.append({
    #             'mal_id': info['mal_id'],
    #             **item,
    #         })

    with TemporaryDirectory() as upload_dir:
        df_animes = pd.DataFrame(anime_records)
        df_animes = df_animes.sort_values(by=['mal_id'], ascending=[False])
        df_animes.to_parquet(os.path.join(upload_dir, 'animes.parquet'))

        df_items = pd.DataFrame(item_records)
        df_items = df_items.sort_values(by=['mal_id', 'published_at'], ascending=[False, False])
        df_items.to_parquet(os.path.join(upload_dir, 'items.parquet'))

        images_dir = os.path.join(upload_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        download_directory_as_directory(
            repo_id=repository,
            repo_type='dataset',
            local_directory=images_dir,
            dir_in_repo='images',
        )

        def _fn_download_poster(aitem):
            poster_url = aitem['poster_url']
            dst_filename = os.path.join(images_dir, aitem['poster_filename'])
            if os.path.exists(dst_filename):
                return

            download_file(
                url=poster_url,
                filename=dst_filename,
                session=session,
            )

        parallel_call(df_animes.to_dict('records'), _fn_download_poster, desc='Download Posters')

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
            print('- erairaws', file=f)
            print('- myanimelist', file=f)
            print('---', file=f)
            print('', file=f)

            print(f'This is the information integration of erai-raws and myanimelist.', file=f)
            print(f'', file=f)

            animes_shown = []
            ext_names = []
            for aitem in df_animes.to_dict('records'):
                ext_names.extend(aitem['external_links'])
            ext_names = sorted(set(ext_names))
            for aitem in df_animes[~df_animes['published_at'].isnull()][:500].replace(np.nan, None).to_dict('records'):
                poster_shown_url = hf_hub_url(
                    repo_id=repository,
                    repo_type='dataset',
                    filename=f'images/{aitem["poster_filename"]}'
                )
                animes_shown.append({
                    'ID': aitem['mal_id'],
                    'Post': f'[![{aitem["id"]}]({poster_shown_url})]({aitem["external_links"]["MAL"]})',
                    'Bangumi': f'[{aitem["title"]}]({aitem["page_url"]})',
                    'RSS': f'[RSS]({aitem["rss_url"]})' if aitem['rss_url'] else '',
                    **{
                        extname: f'[{extname}]({aitem["external_links"][extname]})'
                        if aitem["external_links"].get(extname) else ''
                        for extname in ext_names
                    },
                    'Resources': len(aitem["resources"]),
                    'Published At': (datetime.datetime.fromtimestamp(aitem['published_at']).isoformat()
                                     if aitem['published_at'] else 'N/A'),
                    'Last Published At': (datetime.datetime.fromtimestamp(aitem['last_published_at']).isoformat()
                                          if aitem['last_published_at'] else 'N/A'),
                })
            df_animes_shown = pd.DataFrame(animes_shown)

            print(f'# Animes', file=f)
            print(f'', file=f)
            print(f'{plural_word(len(df_animes), "anime")} in total, '
                  f'{plural_word(len(df_animes_shown), "anime")} shown.', file=f)
            print(f'', file=f)
            print(df_animes_shown.to_markdown(index=False), file=f)
            print(f'', file=f)

            sec_names = []
            res_names = []
            for iitem in df_items.to_dict('records'):
                sec_names.extend(iitem['sec_links'].keys())
                res_names.extend(iitem['resource_urls'].keys())
            sec_names = sorted(set(sec_names))
            res_names = sorted(set(res_names))

            items_shown = []
            for iitem in df_items[:50].to_dict('records'):
                items_shown.append({
                    'Anime ID': iitem['mal_id'],
                    'Title': f'[{iitem["title"]}]({iitem["page_url"]})',
                    'Categories': ", ".join(iitem['categories']),
                    'Langs': ", ".join(iitem['langs']),
                    **{
                        name: (
                            f'[{name}]({_url_safe(iitem["sec_links"][name])})'
                            if iitem["sec_links"].get(name) else ''
                        ) for name in sec_names
                    },
                    **{
                        name: (
                            f'[{name}]({_url_safe(iitem["resource_urls"][name])})'
                            if isinstance(iitem["resource_urls"][name], str) else
                            f'[{name}]({_url_safe(iitem["resource_urls"][name]["torrent"])})'
                        ) if iitem["resource_urls"].get(name) else ''
                        for name in res_names
                    },
                    'Published At': datetime.datetime.fromtimestamp(iitem['published_at']).isoformat(),
                })
            df_items_shown = pd.DataFrame(items_shown)

            print(f'# Resources', file=f)
            print(f'', file=f)
            print(f'{plural_word(len(df_items), "resource")} in total, '
                  f'{plural_word(len(df_items_shown), "resource")} shown.', file=f)
            print(f'', file=f)
            print(df_items_shown.to_markdown(index=False), file=f)
            print(f'', file=f)

        upload_directory_as_directory(
            repo_id=repository,
            repo_type='dataset',
            local_directory=upload_dir,
            path_in_repo='.',
            message=f'Syncing {plural_word(len(df_animes), "anime")} into index',
        )


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    sync(
        repository='deepghs/erairaws_infos',
        proxy_pool=os.environ['PP_SITE'],
    )
