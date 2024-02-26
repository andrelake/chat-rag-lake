from pprint import pprint

from env import PINECONE_API_KEY
from connections.pinecone import get_database_client, get_vectorstore, query_documents


# Get database client
database_client = get_database_client(PINECONE_API_KEY)

# Get vectorstore
vectorstore_name = 'felipe-dev-picpay-prj-ai-rag-llm-pdf'
vectorstore = get_vectorstore(vectorstore_name, database_client)

# Search for similar documents
query = 'O que o governo brasileiro propõe para melhorar a educação?'
result_documents = query_documents(vectorstore, query, k=3)
pprint([{'page_content': r.page_content, 'metadata': r.metadata} for r in result_documents])