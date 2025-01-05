import os
from functools import lru_cache

from openai import OpenAI


@lru_cache()
def get_openai_client():
    return OpenAI(
        base_url=os.environ['LLM_SITE'],
        api_key=os.environ['LLM_API_KEY']
    )
