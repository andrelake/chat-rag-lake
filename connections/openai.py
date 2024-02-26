from typing import Union, Tuple, List, Set

from utils import log

import tiktoken
from langchain_openai import OpenAIEmbeddings


def get_embeddings_client(api_key: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(openai_api_key=api_key)


def get_embedding_cost(documents: Union[Tuple, List, Set]) -> None:
    log(f'Starting embedding price calculation')
    encoding = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(encoding.encode(document.page_content)) for document in documents])
    log(f'Total tokens: {total_tokens}')
    log(f'Total pages: {len(documents)}')
    log(f'Embedding cost: ${total_tokens * 0.0004 / 1000:.4f}')
