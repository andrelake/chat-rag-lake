from typing import Union, Tuple, List, Set

from utils import log

import tiktoken
from langchain_openai import OpenAIEmbeddings


models_pricing = {
    'text-embedding-3-small': 0.00002,
    'text-embedding-3-large': 0.00013,
    'text-embedding-ada-002': 0.00010,
}  # in dollars per token, as of 2024-02-26


def get_embeddings_client(api_key: str, model_name: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(openai_api_key=api_key, model=model_name)


def get_embedding_cost(documents: Union[Tuple, List, Set], model_name: str) -> None:
    log(f'Starting embedding price calculation')
    encoding = tiktoken.encoding_for_model(model_name)
    total_tokens = sum([len(encoding.encode(document.page_content)) for document in documents])
    total_documents = len(documents)
    log(f'Total tokens: {total_tokens}', end='\n')
    log(f'Total documents: {total_documents} (~{total_tokens/total_documents:.1f}/doc)', end='\n')
    log(f'Embedding cost: ${total_tokens * models_pricing[model_name] / 1000:.05f}')
