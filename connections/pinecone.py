from typing import Union, Tuple, List
from time import sleep
import os
import json

from env import PINECONE_ENVIRONMENT
from utils.utils import log

from langchain_community.vectorstores import Pinecone as LangchainPineconeVectorstore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm


def get_database_client(api_key: str) -> Pinecone:
    return Pinecone(api_key=api_key)


def create_vectorstore(name: str, dimension_count: int, embedding_function: OpenAIEmbeddings, database_client: Pinecone) -> LangchainPineconeVectorstore:
    log(f'Creating vectorstore {name}')
    database_client.create_index(
        name,
        dimension=dimension_count,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region=PINECONE_ENVIRONMENT)
    )
    # wait for index to be initialized
    while not database_client.describe_index(name).status['ready']:
        sleep(1)
    vectorstore = get_vectorstore(
        name=name,
        embedding_function=embedding_function,
        database_client=database_client,
        create=False,
        dimension_count=dimension_count
    )
    return vectorstore


def get_vectorstore(name: str, embedding_function: OpenAIEmbeddings, database_client: Pinecone, create: bool = False, dimension_count: int = None) -> LangchainPineconeVectorstore:
    try:
        vectorstore = LangchainPineconeVectorstore.from_existing_index(name, embedding_function)
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


def delete_vectorstore(name: str, database_client: Pinecone) -> None:
    log(f'Deleting vectorstore: {name}')
    if name not in database_client.list_indexes().names():
        log(f'Vectorstore "{name}" not found, skipping deletion')
        return
    database_client.delete_index(name)
    log(f'Successfully deleted vectorstore: {name}.')


def add_documents(vectorstore: LangchainPineconeVectorstore, documents: List[Document], embedding_function: OpenAIEmbeddings, vectorstore_name: str) -> None:
    batch_size = 512
    total_documents = len(documents)
    added_documents = 0
    logs = []
    for i in tqdm(range(0, len(documents), batch_size)):
        i_end = min(i+batch_size, len(documents))
        try:
            vectorstore.from_documents(documents[i:i_end], embedding_function, index_name=vectorstore_name)
            added_documents += i_end - i
        except Exception as e:
            logs.append(f'Error adding documents {i}-{i_end} to vectorstore: {e}')
            error_path = os.path.join('data', 'errors')
            os.makedirs(error_path, exist_ok=True)
            with open(os.path.join(error_path, f'error_{vectorstore_name}_{i}to{i_end}.log'), 'w') as fi:
                fi.write('\n\n'.join([doc.page_content + '\n' + json.dumps(doc.metadata) for doc in documents[i:i_end]]))
    log('\n'.join(logs))
    log(f'Added {added_documents}/{total_documents} documents to vectorstore')


def query_documents(vectorstore: LangchainPineconeVectorstore, query: Union[str, List[str]], k: int = 10) -> Tuple[List[str], List[float]]:
    log(f'Querying vectorstore with {query}')
    return vectorstore.similarity_search(query, k=k)