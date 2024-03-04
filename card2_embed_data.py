import os
from env import PINECONE_API_KEY, OPENAI_API_KEY
from datetime import date
from typing import Optional, Any, Callable, List, Dict

from utils import log, get_month_name, threat_product, threat_card_variant
from data_handler import read_orc, generate_documents, redistribute_by_characters, get_embeddings_client, get_embedding_cost
from data_tables import CardTransactions
from connections.pinecone import (
    get_database_client,
    create_vectorstore,
    get_vectorstore,
    delete_vectorstore,
    add_documents,
    query_documents,
)

from langchain_core.documents import Document


# Configure Logger
log.verbose = True
log.end = '\n\n'

# Load data
df = CardTransactions.read()

# Get database client
database_client = get_database_client(api_key=PINECONE_API_KEY)

# Get embeddings client
embedding_model_name = 'text-embedding-3-small'
embedding_function = get_embeddings_client(model_name=embedding_model_name, type='api', api_key=OPENAI_API_KEY)

# Generate documents
## By portfolio, year, month, day, portfolio, consumer, and transaction
documents = generate_documents(
    df=df,
    where=None,
    group_by=None,
    order_by=['transaction_year', 'transaction_month', 'transaction_day', 'portfolio_id', 'consumer_id', 'transaction_at'],
    limit=None,
    parse_content_header=lambda record:
        f'''O cliente {record['consumer_name']} (CPF: {record['consumer_document']}), '''
        f'''que pertence à carteira do gerente de contas {record['officer_name']} (ID {record['officer_id']}), '''
        f'''efetuou um transação de R$ {record['transaction_value']:.2f} '''
        f'''no dia {record['transaction_day']} do mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
        f'''({record['transaction_day']:02}/{record['transaction_month']:02}/{record['transaction_year']:04}) '''
        f'''com cartão de {threat_product(record['product'])} {threat_card_variant(record['card_variant'])} para o estabelecimento "{record['seller_description']}"''',
    parse_content_body=None,
    parse_metadata=lambda record: dict(record)
)


def insert_documents(vectorstore_name: str, documents: List[Document]):
    # Get or create vectorstore
    delete_vectorstore(vectorstore_name, database_client)
    vectorstore = get_vectorstore(
        name=vectorstore_name,
        embedding_function=embedding_function,
        database_client=database_client,
        create=True,
        dimension_count=1536
    )

    # Add documents
    get_embedding_cost(documents=documents, model_name=embedding_model_name)  # If using OpenAI Embeddings
    add_documents(
        vectorstore=vectorstore,
        documents=documents,
        embedding_function=embedding_function,
        vectorstore_name=vectorstore_name,
    )


# Storytelling for each transaction, chunked by 1000 characters
def test_1():
    vectorstore_name = 'felipe-dev-picpay-prj-ai-rag-llm-table-1'
    documents_test_1 = redistribute_by_characters(documents, chunk_size=1000, chunk_overlap=50)
    insert_documents(vectorstore_name, documents_test_1)


# Storytelling for each transaction, chunked by transaction
def test_2():
    # Get or create vectorstore
    vectorstore_name = 'felipe-dev-picpay-prj-ai-rag-llm-table-2'
    documents_test_2 = documents
    insert_documents(vectorstore_name, documents_test_2)


# Storytelling for each unit in each level of aggregations, mixed chunks scopes
def test_3():
    pass


test_1()
test_2()
test_3()
