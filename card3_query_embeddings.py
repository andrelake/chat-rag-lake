from pprint import pprint

from utils import log


log.verbose = True
log.end = '\n\n'


from env import PINECONE_API_KEY, OPENAI_API_KEY
from connections.openai import get_embeddings_client, get_self_query_retriever
from connections.pinecone import get_database_client, get_vectorstore, query_documents
from data_handler.avro import get_documents_metadata


# Get database client
database_client = get_database_client(PINECONE_API_KEY)

# Get embeddings client
embedding_model_name = 'text-embedding-3-small'
embedding_function = get_embeddings_client(OPENAI_API_KEY, model_name='text-embedding-3-small')  # API, OpenAI, 1536 dimensions

# Get vectorstore
vectorstore_name = 'felipe-dev-picpay-prj-ai-rag-llm-table-1'
vectorstore = get_vectorstore(
    name=vectorstore_name,
    embedding_function=embedding_function,
    database_client=database_client,
    create=False,
    dimension_count=1536
)

# Get documents metadata
documents_description, documents_metadata = get_documents_metadata()
retriever = get_self_query_retriever(vectorstore, documents_description, documents_metadata)

# Ask user for prompt until user types "exit"
while True:
    # Ask for prompt
    prompt = input("Enter a prompt (type 'exit' to quit): ")
    if prompt == "exit":
        break
    response = retriever.invoke(prompt)
    print(response)

# Search for similar documents
# prompt = "O cliente de CPF 149.206.583-89 efetuou quais transaçoes em março de 2023?"
# result_documents = query_documents(vectorstore, query, k=3)
# for document in result_documents:
#     pprint(document.metadata)
#     print(document.page_content, end='\n\n')
