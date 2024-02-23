from typing import Union, Tuple, List, Set
import re
from env import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Pinecone as PineconeVectorstore
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


def chunk_data(data: List, chunk_size: int = 1600, chunk_overlap: int = 0) -> List:
    log(f'Starting chunk context')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(data)
    for i, chunk in enumerate(chunks):
        log(f'#{i+1} {chunk.page_content}')
    log(f'Total chunks: {len(chunks)}')
    return chunks


def insert_or_fetch_embeddings(index_name: str, chunks: List, pinecone: Pinecone, embeddings: OpenAIEmbeddings) -> PineconeVectorstore:
    indexes = pinecone.list_indexes()
    log(f'Indexes: {len(indexes)}')
    if index_name in indexes.names():
        log(f'Getting info from index: {index_name}')
        return PineconeVectorstore.from_existing_index(index_name, embeddings)
    else:
        return create_vector_store(chunks, embeddings, index_name, pinecone)


def create_vector_store(chunks: List, embeddings: OpenAIEmbeddings, index_name: str, pinecone: Pinecone) -> PineconeVectorstore:
    log(f'Creating index: {index_name}')
    pinecone.create_index(
        index_name,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region=PINECONE_ENVIRONMENT)
    )
    return PineconeVectorstore.from_documents(chunks, embeddings, index_name=index_name)


def delete_pinecone_index(index_name: str, pinecone: Pinecone) -> None:
    log(f'Deleting index: {index_name}"')
    pinecone.delete_index(index_name)
    log(f'Index Deleted Successfully.')


def show_embeddings_cost(texts: Union[Tuple, List, Set]) -> None:
    log(f'Starting embedding price calculation')
    encoding = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(encoding.encode(page.page_content)) for page in texts])
    log(f'Total tokens: {total_tokens}')
    log(f'Total pages: {len(texts)}')
    log(f'Embedding cost: ${total_tokens * 0.0004 / 1000:.4f}')
