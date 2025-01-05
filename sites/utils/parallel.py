import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Callable, Any, Optional

from tqdm import tqdm


def parallel_call(iterable: Iterable, fn: Callable[[Any], None], total: Optional[int] = None,
                  desc: Optional[str] = None, max_workers: Optional[int] = None):
    if total is None:
        try:
            total = len(iterable)
        except (TypeError, AttributeError):
            total = None

    pg = tqdm(total=total, desc=desc or f'Process with {fn!r}')
    if not max_workers:
        max_workers = min(os.cpu_count(), 16)
    tp = ThreadPoolExecutor(max_workers=max_workers)

    def _fn(item):
        try:
            return fn(item)
        except Exception as err:
            logging.exception(f'Error when processing {item!r} - {err!r}')
            raise
        finally:
            pg.update()

    for item in iterable:
        tp.submit(_fn, item)

    tp.shutdown(wait=True)
