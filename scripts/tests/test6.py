'''
Test 6 - Chunks of 1000 tokens
'''


from env import PINECONE_API_KEY, OPENAI_API_KEY
from typing import List

from utils import log, get_month_name
from data_handler import DocumentsHandler
from data_tables import CardTransactions
from connections.embeddings import get_client, get_cost, get_dimension_count
from connections.pinecone import (
    get_database_client,
    get_vectorstore,
    delete_vectorstore,
    add_documents,
)

from pandas import concat
from langchain_core.documents import Document


# Configure Logger
log.verbose = True
log.end = '\n\n'


class Embedding:
    model_name = 'text-embedding-3-small'
    type = 'api'
    client = get_client(model_name=model_name, type=type, api_key=OPENAI_API_KEY)
    dimension_count = get_dimension_count(model_name=model_name, type=type)


class Database:
    client = get_database_client(PINECONE_API_KEY)
    vectorstore_name = 'prj-ai-rag-llm-table-6-chunks'
    vectorstore = get_vectorstore(
        name=vectorstore_name,
        embedding_function=Embedding.client,
        database_client=client,
        create=True,
        dimension_count=Embedding.dimension_count
    )


from test5 import documents


documents = DocumentsHandler.redistribute_by_characters(documents, 1000, 0)

if __name__ == '__main__':
    DocumentsHandler.write_txt(Database.vectorstore_name, documents)

    # Add documents
    delete_vectorstore(name=Database.vectorstore_name, database_client=Database.client)
    get_cost(documents=documents, model_name=Embedding.model_name)
    add_documents(
        vectorstore=Database.vectorstore,
        documents=documents,
        embedding_function=Embedding.client,
        vectorstore_name=Database.vectorstore_name,
    )
    log(Database.vectorstore._index.describe_index_stats())
