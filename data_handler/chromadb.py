import os
from typing import Optional, Any, Callable, Union, Tuple, List, Set
import re
from env import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma as LangchainChromaVectorstore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
import chromadb
import fastavro
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


def delete_collection(name: str, chroma_client: chromadb.Client) -> None:
    log(f'Deleting collection: {name}')
    chroma_client.delete_collection(name)
    log(f'Successfully Deleted Collection: {name}.')


def add_documents(langchain_db_collection: LangchainChromaVectorstore, documents: List) -> None:
    langchain_db_collection._collection.add(documents)
    log(f'Added {len(documents)} documents to collection')


def query_collection(langchain_db_collection: LangchainChromaVectorstore, query: str) -> None:
    log(f'Querying collection')
    result = langchain_db_collection.similarity_search(query)
    log(f'Result: {result}')
    return result


def show_embeddings_cost(documents: Union[Tuple, List, Set]) -> None:
    log(f'Starting embedding price calculation')
    encoding = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(encoding.encode(document.page_content)) for document in documents])
    log(f'Total tokens: {total_tokens}')
    log(f'Total pages: {len(documents)}')
    log(f'Embedding cost: ${total_tokens * 0.0004 / 1000:.4f}')

def get_month_name(n: int) -> str:
    return ('janeiro','fevereiro','março','abril','maio','junho','julho','agosto','setembro','outubro','novembro','dezembro')[n-1]

def extract_documents_from_file(
    file_path: str,
    group_by: str,
    group_body: Callable[[Any], str],
    aggregated_body: Callable[[Any], str],
    filter: Optional[Callable[[Any], bool]] = None
) -> List[Document]:  # TODO: Test pandas implementation
    document: Document = None
    documents: List[Document] = list()
    stream_last_group_headers = None
    stream_current_group_headers = None
    count_processed_rows: int = 0
    count_total_rows: int = 0

    log(f'Reading data from `{file_path}`')
    with open(file_path, 'rb') as fp:
        for block in fastavro.block_reader(fp):
            count_total_rows += block.num_records
            try:
                for record in block:
                    if not filter(record):
                        continue
                    stream_current_group_headers = tuple(record[header] for header in group_by)
                    if stream_current_group_headers != stream_last_group_headers:
                        if document:
                            documents.append(document)
                        document = Document(
                            page_content=group_body(record),
                            metadata={'transaction_id': record['transaction_id']}
                        )
                        stream_last_group_headers = stream_current_group_headers
                    document.page_content += aggregated_body(record)
                    count_processed_rows += 1
            except Exception as e:
                log(f'Error: {e}. Skipping record.', end='\n')
                continue
        if document:
            documents.append(document)
    log(f'Processed {count_processed_rows:_}/{count_total_rows:_} rows from `{file_path}` into {len(documents)} documents.')
    return documents


def extract_documents(
    path: str,
    group_by: str,
    group_body: Callable[[Any], str],
    aggregated_body: Callable[[Any], str],
    filter: Optional[Callable[[Any], bool]] = None
) -> List[Document]:  # TODO: Test pandas implementation
    
    log(f'Extracting documents from `{path}`')
    source_files = []
    # Example: filepath = 'data/card_transactions/'
    # Real file: 'data/card_transactions/transaction_year=2023/part-00002-tid-1056622170898768028-a56bd3a2-d283-4b7a-ab87-62442d905a78-116499-1.c000.avro'
    # Example: files = ['data/card_transactions/transaction_year=2023/part-00002-tid-1056622170898768028-a56bd3a2-d283-4b7a-ab87-62442d905a78-116499-1.c000.avro']
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.avro'):
                file_path = os.path.join(root, file)
                source_files.append(file_path)
                log(file_path, end='\n')

    documents = []
    for file_path in source_files:
        documents.extend(
            extract_documents_from_file(
                file_path=file_path,
                group_by=group_by,
                group_body=group_body,
                aggregated_body=aggregated_body,
                filter=filter
            )
        )
    return documents
