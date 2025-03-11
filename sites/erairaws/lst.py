import datetime
import os.path
from typing import Optional

import dateparser
import pandas as pd
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory, urlsplit
from hfutils.cache import delete_detached_cache
from hfutils.operate import download_directory_as_directory, get_hf_client, get_hf_fs, upload_directory_as_directory
from hfutils.utils import number_to_tag
from huggingface_hub import hf_hub_url
from tqdm import tqdm

from .info import get_session, iter_anime_items, get_anime_info
from ..utils import parallel_call, download_file


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
    session_rss = get_session(no_login=True)
    if proxy_pool:
        logging.info(f'Proxy pool {proxy_pool!r} enabled.')
        session.proxies.update({
            'http': proxy_pool,
            'https': proxy_pool
        })
        session_rss.proxies.update({
            'http': proxy_pool,
            'https': proxy_pool
        })

    anime_items = list(iter_anime_items(session=session))
    anime_records = []
    item_records = []
    for title, page_url in tqdm(anime_items[:100], desc='Animes'):
        logging.info(f'Processing {title!r}, page: {page_url!r} ...')

        info = get_anime_info(page_url, session=session, session_rss=session_rss)
        if not info['mal_id']:
            logging.warning(f'No MAL ID found for {page_url!r}, skipped.')
            continue

        _, ext = os.path.splitext(urlsplit(info['poster_url']).filename)
        anime_records.append({
            **info,
            'poster_filename': f'{info["id"]}{ext}',
            'last_published_at': max([
                dateparser.parse(item['pubDate']).timestamp()
                for item in info['rss_items']
            ]) if info['rss_items'] else None,
        })
        for item in info['rss_items']:
            item_records.append({
                'mal_id': info['mal_id'],
                **item,
                'published_at': dateparser.parse(item['pubDate']).timestamp(),
            })

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
            for aitem in df_animes.to_dict('records'):
                poster_shown_url = hf_hub_url(
                    repo_id=repository,
                    repo_type='dataset',
                    filename=f'images/{aitem["poster_filename"]}'
                )
                animes_shown.append({
                    'ID': aitem['mal_id'],
                    'Post': f'[![{aitem["id"]}]({poster_shown_url})]({aitem["external_links"]["MAL"]})',
                    'Bangumi': f'[{aitem["title"]}]({aitem["page_url"]})',
                    'Resolution': aitem["rss_items_res"],
                    'RSS': f'[RSS]({aitem["rss_url"]})' if aitem['rss_url'] else '',
                    **{
                        extname: f'[{extname}]({aitem["external_links"][extname]})' if aitem["external_links"].get(
                            extname) else ''
                        for extname in ext_names
                    },
                    'Magnets': len(aitem["rss_items"]),
                    'Last Published At': datetime.datetime.fromtimestamp(aitem['last_published_at']).isoformat(),
                })
            df_animes_shown = pd.DataFrame(animes_shown)

            print(f'# Animes', file=f)
            print(f'', file=f)
            print(f'{plural_word(len(df_animes), "anime")} in total.', file=f)
            print(f'', file=f)
            print(df_animes_shown.to_markdown(index=False), file=f)
            print(f'', file=f)

            print(f'# Resources', file=f)
            print(f'', file=f)
            print(f'{plural_word(len(df_items), "resources")} in total.', file=f)
            print(f'', file=f)
            items_shown = []
            for iitem in df_items[:50].to_dict('records'):
                items_shown.append({
                    'Anime ID': iitem['mal_id'],
                    'Title': iitem['title'],
                    'Resolution': iitem['erai:resolution'],
                    'Category': iitem['erai:category'],
                    'Size': iitem['erai:size'],
                    'InfoHash': iitem['erai:infohash'],
                    'Subtitles': iitem['erai:subtitles'],
                    'Published At': iitem['pubDate'],
                })
            df_items_shown = pd.DataFrame(items_shown)
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
