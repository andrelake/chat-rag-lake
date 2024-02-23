from typing import Union, Tuple, List, Set
import re
from env import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma as LangchainChromaVectorstore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
import chromadb
import tiktoken


class Logger:
    def __init__(self, end: str = '\n', verbose: bool = False, **kwargs):
        self.end = end
        self.verbose = verbose
        self.kwargs = kwargs

    def __call__(self, message: str, **kwargs):
        kwargs = {'end': self.end, **self.kwargs, **kwargs}
        if self.verbose:
            print(message, **kwargs)


log = Logger()


def get_embeddings_client(api_key: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(openai_api_key=api_key)


def get_chromadb_client(host: str, port: int) -> chromadb.HttpClient:
    return chromadb.HttpClient(
        host=host,
        port=port,
        settings=chromadb.config.Settings(allow_reset=True, anonymized_telemetry=False)
    )


def get_collection(name: str, embedding_function: OpenAIEmbeddings, db_client: chromadb.Client) -> LangchainChromaVectorstore:
    log(f'Getting collection: {name}')
    langchain_db_collection = LangchainChromaVectorstore(
        client=db_client,
        collection_name=name,
        embedding_function=embedding_function,
    )
    return langchain_db_collection


def create_collection(name: str, embedding_function: OpenAIEmbeddings, db_client: chromadb.Client) -> LangchainChromaVectorstore:
    log(f'Creating collection')
    langchain_db_collection = get_collection(name, embedding_function, db_client)
    return langchain_db_collection


def get_or_create_collection(name: str, embedding_function: OpenAIEmbeddings, db_client: chromadb.Client) -> LangchainChromaVectorstore:
    log(f'Getting/Creating collection')
    langchain_db_collection = get_collection(name, embedding_function, db_client)
    return langchain_db_collection


def insert_or_fetch_embeddings(index_name: str, chunks: List, pinecone: Pinecone, embeddings: OpenAIEmbeddings) -> PineconeVectorstore:
    indexes = pinecone.list_indexes()
    log(f'Indexes: {len(indexes)}')
    if index_name in indexes.names():
        log(f'Getting info from index: {index_name}')
        return PineconeVectorstore.from_existing_index(index_name, embeddings)
    else:
        return create_vector_store(chunks, embeddings, index_name, pinecone)


def show_embeddings_cost(texts: Union[Tuple, List, Set]) -> None:
    log(f'Starting embedding price calculation')
    encoding = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(encoding.encode(page.page_content)) for page in texts])
    log(f'Total tokens: {total_tokens}')
    log(f'Total pages: {len(texts)}')
    log(f'Embedding cost: ${total_tokens * 0.0004 / 1000:.4f}')
