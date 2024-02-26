from pprint import pprint

from env import PINECONE_API_KEY, OPENAI_API_KEY
from connections.openai import get_embeddings_client
from connections.pinecone import get_database_client, get_vectorstore, query_documents


# Get database client
database_client = get_database_client(PINECONE_API_KEY)

# Get embeddings client
embedding_model_name = 'text-embedding-3-small'
embedding_function = get_embeddings_client(OPENAI_API_KEY, model_name='text-embedding-3-small')  # API, OpenAI, 1536 dimensions

# Get vectorstore
vectorstore_name = 'felipe-dev-picpay-prj-ai-rag-llm-table'
vectorstore = get_vectorstore(
    name=vectorstore_name,
    embedding_function=embedding_function,
    database_client=database_client,
    create=False,
    dimension_count=1536
)

# Search for similar documents
query = "O cliente Vitor Gabriel Gomes de CPF 452.830.971-88 efetuou quais transaçoes em março de 2024?"
result_documents = query_documents(vectorstore, query, k=3)
for document in result_documents:
    pprint(document.metadata)
    print(document.page_content, end='\n\n')
