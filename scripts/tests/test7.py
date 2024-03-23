'''
Test 7 - Cohere embeddings
'''


from env import OPENAI_API_KEY
from utils import log
from data_utils.handlers import DocumentsHandler
from connections.embeddings import get_client, get_cost, get_dimension_count
from connections.pinecone import (
    get_vectorstore,
    delete_vectorstore,
    add_documents,
)


# Configure Logger
log.verbose = True
log.end = '\n\n'


class Embedding:
    model_name = 'cohere...'
    type = 'api'
    client = get_client(model_name=model_name, type=type, api_key=OPENAI_API_KEY)
    dimension_count = get_dimension_count(model_name=model_name, type=type)


class Database:
    client = CohereClient()
    vectorstore_name = 'prj-ai-rag-llm-table-7-cohere'
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
