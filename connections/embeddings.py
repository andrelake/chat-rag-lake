from typing import Callable, List

from utils import log
from chromadb.utils import embedding_functions as _chromadb_embedding_functions

import tiktoken
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.globals import set_verbose, set_debug


set_debug(log.verbose)
set_verbose(log.verbose)


models = {
    # Local (CPU)
    'cpu': {
        'all-MiniLM-L6-v2': {
            'cost': 0,
            'dimension_count': 384,
            'client': lambda x: _chromadb_embedding_functions.SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2'),
        },
    },
    # Local (GPU)
    'gpu': {
        'all-MiniLM-L6-v2': {
            'cost': 0,
            'dimension_count': 384,
            'client': lambda x: _chromadb_embedding_functions.ONNXMiniLM_L6_V2(preferred_providers=['DmlExecutionProvider']),
        },
    },
    # Managed API
    'api': {
        'text-embedding-3-small': {
            'cost': 0.00002,
            'dimension_count': 1536,
            'client': lambda x: OpenAIEmbeddings(openai_api_key=x['api_key'], model='text-embedding-3-small'),
        },
        'text-embedding-3-large': {
            'cost': 0.00013,
            'dimension_count': 3072,
            'client': lambda x: OpenAIEmbeddings(openai_api_key=x['api_key'], model='text-embedding-3-large'),
        },
        'text-embedding-ada-002': {
            'cost': 0.00010,
            'dimension_count': 1536,
            'client': lambda x: OpenAIEmbeddings(openai_api_key=x['api_key'], model='text-embedding-ada-002'),
        },
        'embed-english-v3': {
            'cost': 0.00010,
            'dimension_count': 1024,
            'client': None,  # TODO: Implement Cohere models client
        },
    }
}  # in dollars per 1k tokens. Updated at 2024-02-26


def list_clients() -> None:
    # Return everything in models except for the 'client' key
    return {
        tk: {
            mk: {
                pk: pv for pk, pv in tv.items() if pk != 'client'
            } for mk, tv in tv.items()
        } for tk, tv in models.items()
    }


def get_client(model_name: str, type: str = 'api', api_key: str = None) -> Callable:
    return models[type][model_name]['client']({'api_key': api_key})


def get_cost(documents: List[Document], model_name: str, type: str) -> None:
    log(f'Calculating embedding cost for {len(documents)} documents using model `{model_name}`...')
    encoding = tiktoken.encoding_for_model(model_name)
    token_count = sum([len(encoding.encode(document.page_content)) for document in documents])
    document_count = len(documents)
    cost = {
        'model_name': model_name,
        'document_count': document_count,
        'token_count': token_count,
        'total_cost': token_count * models[type][model_name]['cost'] / 1000
    }
    log(f'Total tokens: {cost["token_count"]}\n'
        f'Total documents: {cost["document_count"]} (~{cost["token_count"]/cost["document_count"]:.1f} tokens/document)\n'
        f'Total embedding cost: ${cost["total_cost"]:.05f}')
    return cost


def get_dimension_count(model_name: str, type: str = 'api') -> int:
    return models[type][model_name]['dimension_count']