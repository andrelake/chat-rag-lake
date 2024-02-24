import os
from env import CHROMA_DB_HOST, CHROMA_DB_PORT, OPENAI_API_KEY
from datetime import date

from connections.chromadb import (
    log,
    get_chromadb_client,
    get_embeddings_client,
    extract_documents,
    create_collection,
    add_documents,
    query_collection,
    delete_collection,
    show_embeddings_cost,
    get_month_name
)
from data_handler import generate_dummy_data, validation_quiz
from chromadb.utils import embedding_functions


def threat_product(product):
    return {'debit': 'débito', 'credit': 'crédito', '': 'desconhecido'}[product or '']


# Configure Logger
log.verbose = True
log.end = '\n\n'

avro_path = os.path.join('data', 'card_transactions')

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
print(df)
validation_quiz(df, log)

# Load document and chunk data
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
            f'''às {record['transaction_at'].strftime('%H:%M')} de R$ {record['transaction_value']} '''
            f'''para '{record['seller_description'].strip()}\'.''',
    filter=
        lambda record:
            record['transaction_year'] == 2024 and record['portfolio_id'] == 39 and record['transaction_month'] == 2 and record['transaction_day'] >= 19
)
show_embeddings_cost(documents)

# ChromaDB vectorstore client
db_client = get_chromadb_client(host=CHROMA_DB_HOST, port=CHROMA_DB_PORT)

# OpenAI embeddings client
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
                     # get_embeddings_client(OPENAI_API_KEY)  # OpenAI
                     # embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")  # Local Open Source

# Create collection
langchain_db_collection_name = 'felipe-dev-picpay-prj-ai-rag-llm'
try:
    langchain_db_collection = delete_collection(name=langchain_db_collection_name, chroma_client=db_client)
except:
    pass
for batch in range(0, len(documents), 200):
    add_documents(
        create_collection(
            name=langchain_db_collection_name,
            embedding_function=embedding_function,
            db_client=db_client
        ),
        documents[batch:batch+200]
    )

# Add documents
add_documents(langchain_db_collection, documents[:10])

# Search for similar documents
query = "Qual o total do valor de transações do gerente Bruno Pessolato em janeiro de 2024?"
result_documents = query_collection(langchain_db_collection, query)

# Show result
log(result_documents)