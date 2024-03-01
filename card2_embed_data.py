import os
from env import PINECONE_API_KEY, OPENAI_API_KEY
from datetime import date

from utils import log, get_month_name, threat_product

from chromadb.utils import embedding_functions
from connections.pinecone import (
    get_database_client,
    create_vectorstore,
    get_vectorstore,
    delete_vectorstore,
    add_documents,
    query_documents,
)
from connections.openai import get_embeddings_client, get_embedding_cost
from data_handler import read_orc, generate_documents


# Configure Logger
log.verbose = True
log.end = '\n\n'

# Load data
df = read_orc(path=os.path.join('data', 'landing', 'card_transactions.orc'), log=log)

# Generate documents
## By portfolio, year, month, and consumer
documents = generate_documents(
    df=df,
    where=None,
    group_by=None,
    order_by=['portfolio_id', 'transaction_year', 'transaction_month', 'consumer_id', 'transaction_at'],
    limit=None,
    parse_content_header=lambda record: f'''O cliente {record['consumer_name']} (CPF: {record['consumer_document']}), '''
                                        f'''que pertence à carteira do gerente de contas {record['manager_name']} (ID {record['manager_id']}), '''
                                        f'''efetuou um transação de R$ {record['transaction_value']:.2f} '''
                                        f'''no dia {record['transaction_day']}/{record['transaction_month']}/{record['transaction_year']} (dd/MM/yyyy) '''
                                        f'''com cartão de {threat_product(record['product'])} {record['card_variant']} para o estabelecimento "{record['seller_description']}"''',
    parse_content_body=None,
    parse_metadata=lambda record: dict(record)
)

# Get database client
database_client = get_database_client(api_key=PINECONE_API_KEY)

# Get embeddings client
embedding_model_name = 'text-embedding-3-small'
embedding_function = (
    get_embeddings_client(OPENAI_API_KEY, model_name='text-embedding-3-small')  # API, OpenAI, 1536 dimensions
    #embedding_functions.ONNXMiniLM_L6_V2(preferred_providers=['DmlExecutionProvider'])  # Local (GPU), Open Source, 384 dimensions
    #embedding_functions.SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2')  # Local, Open Source, 384 dimensions
)

# Get or create vectorstore
vectorstore_name = 'felipe-dev-picpay-prj-ai-rag-llm-table-1'
delete_vectorstore(vectorstore_name, database_client)
vectorstore = get_vectorstore(
    name=vectorstore_name,
    embedding_function=embedding_function,
    database_client=database_client,
    create=True,
    dimension_count=1536  # 384 for all-MiniLM-L6-v2
)

# Add documents
get_embedding_cost(documents=documents, model_name='text-embedding-3-small')  # If using OpenAI Embeddings
add_documents(
    vectorstore=vectorstore,
    documents=documents,
    embedding_function=embedding_function,
    vectorstore_name=vectorstore_name,
)
