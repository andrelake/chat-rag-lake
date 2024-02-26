import os
from env import PINECONE_API_KEY, OPENAI_API_KEY
from datetime import date

from utils import log, get_month_name

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
from data_handler.avro import (
    generate_dummy_data,
    extract_documents,
    validation_quiz
)


def threat_product(product):
    return {'debit': 'débito', 'credit': 'crédito', '': 'desconhecido'}[product or '']


# Configure Logger
log.verbose = True
log.end = '\n\n'

avro_path = os.path.join('data', 'landing', 'card_transactions.avro')

# Generate dummy data
df = generate_dummy_data(
    group_by=[
        'transaction_year',
        'portfolio_id',
        'consumer_id',
    ],
    n_officers=1,
    n_consumers_officer=10,
    n_transactions_consumer_day=6,
    start_date=date(2023, 1, 1),
    end_date=date(2024, 2, 29),
    chaos_consumers_officer=0.5,
    chaos_transactions_client_day=0.5,
    log=log,
    save_path=avro_path
)
validation_quiz(df, log)

# Load documents and chunk data
documents = extract_documents(
    path=avro_path,
    group_by=[
        'transaction_year',
        'portfolio_id',
        'consumer_id',
    ],
    group_body=
        lambda record:
            f'''Durante {record['transaction_month']:02}/{record['transaction_year']:04} (mês de {get_month_name(record['transaction_month'])} '''
            f'''do ano de {record['transaction_year']:04}), o cliente '{record['consumer_name']}' (CPF {record['consumer_document']}) que pertence à carteira de clientes '''
            f'''de ID '{record['portfolio_id']}' do gerente de contas '{record['officer_name']}' efetuou as seguintes transações de cartão variante {record['card_variant']} '''
            f'''na modalidade '{threat_product(record['product'])}':''',
    aggregated_body=
        lambda record:
            f'''\nDia {record['transaction_day']:02}/{record['transaction_month']:02}/{record['transaction_year']:04} '''
            f'''às {record['transaction_at']} de R$ {record['transaction_value']} '''
            f'''para '{record['seller_description'].strip()}\'.''',
    filter=lambda record: True
)

# Get vectorstore client
database_client = get_database_client(api_key=PINECONE_API_KEY)

# OpenAI embeddings client
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")  # Local Open Source
                     # get_embeddings_client(OPENAI_API_KEY)  # OpenAI Embeddings

# Get or create vectorstore
vectorstore_name = 'felipe-dev-picpay-prj-ai-rag-llm-table'
vectorstore = get_vectorstore(vectorstore_name, embedding_function, database_client, create=True, dimension_count=None)

# Add documents
get_embedding_cost(documents)
add_documents(vectorstore, documents[:5])
