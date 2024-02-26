from typing import Union, Tuple, List, Set
from time import sleep

from env import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT
from utils import log

from langchain_community.vectorstores import Pinecone as LangchainPineconeVectorstore
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
    database_client.delete_index(name)
    log(f'Successfully deleted vectorstore: {name}.')


def add_documents(vectorstore: LangchainPineconeVectorstore, documents: List) -> None:
    batch_size = 64
    for i in tqdm(range(0, len(documents), batch_size)):
        i_end = min(i+batch_size, len(documents))
        kargs = [
            {
                'texts': document.page_content,
                'metadatas': document.metadata,
                'ids': f'{document.metadata["transaction_id"]:19}'
            }
            for document in documents[i:i_end]
        ]
        kargs = {key: [item[key] for item in kargs] for key in kargs[0]}
        vectorstore.add_texts(**kargs)
    log(f'Added {len(documents)} documents to vectorstore')


def query_documents(vectorstore: LangchainPineconeVectorstore, query: Union[str, List[str]], k: int = 10) -> Tuple[List[str], List[float]]:
    log(f'Querying vectorstore with {query}')
    return vectorstore.similarity_search(query, k=k)