from pprint import pprint

from utils import log


log.verbose = True
log.end = '\n\n'


from env import PINECONE_API_KEY, OPENAI_API_KEY
from connections.openai import get_self_query_retriever
from connections.pinecone import get_database_client, get_vectorstore, query_documents
from data_tables import CardTransactions
from data_handler import get_embeddings_client


# Get database client
database_client = get_database_client(PINECONE_API_KEY)

# Get embeddings client
embedding_model_name = 'text-embedding-3-small'
embedding_function = get_embeddings_client(model_name=embedding_model_name, type='api', api_key=OPENAI_API_KEY)

# Get vectorstore
vectorstore_name = 'felipe-dev-picpay-prj-ai-rag-llm-table-1'
vectorstore = get_vectorstore(
    name=vectorstore_name,
    embedding_function=embedding_function,
    database_client=database_client,
    create=False,
    dimension_count=1536
)

# Ask user for prompt until user types "exit"
while True:
    # Example: "O cliente Carlos Eduardo Rodrigues (CPF: 80763159212) efetuou quais transaçoes em março de 2023?"
    prompt = input("\033[96mEnter a prompt (type 'exit' to quit):\033[0m ")
    if prompt == "exit":
        break

    # Search for similar documents
    result_documents = query_documents(vectorstore, prompt, k=15)
    for document in result_documents:
        # pprint(document.metadata)
        print(document.page_content, end='\n\n')
