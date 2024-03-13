from typing import Union, Tuple, List

from utils.utils import log

from langchain_community.vectorstores import Chroma as LangchainChromaVectorstore
from langchain_openai import OpenAIEmbeddings
import chromadb


def get_database_client(host: str, port: str) -> chromadb.HttpClient:
    log(f'Getting ChromaDB client at {host}:{port}')
    return chromadb.HttpClient(
        host=host,
        port=str(port),
        settings=chromadb.config.Settings(allow_reset=True, anonymized_telemetry=False)
    )


def create_vectorstore(name: str, dimension_count: int, embedding_function: OpenAIEmbeddings, database_client: chromadb.Client) -> LangchainChromaVectorstore:
    log(f'Creating vectorstore {name}')
    vectorstore = database_client.create_collection(name)
    # vectorstore = get_vectorstore(name, embedding_function, database_client)
    return vectorstore


def get_vectorstore(name: str, embedding_function: OpenAIEmbeddings, database_client: chromadb.Client, create: bool = False, dimension_count: int = None) -> LangchainChromaVectorstore:
    # vectorstore = LangchainChromaVectorstore(
    #     client=database_client,
    #     collection_name=name,
    #     embedding_function=embedding_function,
    # )
    try:
        vectorstore = database_client.get_collection(name)
        log(f'Found vectorstore: "{name}"')
    except:
        if create:
            log(f'Vectorstore "{name}" not found, creating')
            vectorstore = create_vectorstore(
                name=name,
                dimension_count=dimension_count,
                embedding_function=embedding_function,
                database_client=database_client
            )
        else:
            raise
    return vectorstore


def delete_vectorstore(name: str, database_client: chromadb.Client) -> None:
    log(f'Deleting vectorstore: {name}')
    database_client.delete_collection(name)
    log(f'Successfully deleted vectorstore: {name}.')


def add_documents(vectorstore: LangchainChromaVectorstore, documents: List) -> None:
    kargs = [
        {
            'documents': document.page_content[:25],
            'metadatas': document.metadata,
            'ids': f'{document.metadata["transaction_id"]:19}'
        }
        for document in documents
    ]
    kargs = {key: [item[key] for item in kargs] for key in kargs[0]}
    vectorstore.add(**kargs)
    # vectorstore._collection.add(documents)
    log(f'Added {len(documents)} documents to vectorstore')


def query_documents(vectorstore: LangchainChromaVectorstore, query: Union[str, List[str]], k: int = 10) -> Tuple[List[str], List[float]]:
    log(f'Querying vectorstore with {query}')
    results = vectorstore.query(query, k=k)
    # results = vectorstore.similarity_search(query)
    return results
