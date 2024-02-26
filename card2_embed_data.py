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
from data_handler.avro import (
    generate_dummy_data,
    extract_documents,
    validation_quiz
)


# Configure Logger
log.verbose = True
log.end = '\n\n'

# Load documents and chunk data
avro_path = os.path.join('data', 'landing', 'card_transactions.avro')
documents = extract_documents(
    path=avro_path,
    group_by=[
        'transaction_year',
        'portfolio_id',
        'consumer_id',
        'transaction_month',
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

# Get database client
database_client = get_database_client(api_key=PINECONE_API_KEY)

# Get embeddings client
embedding_model_name = 'text-embedding-3-small'
embedding_function_callables = [
    get_embeddings_client(OPENAI_API_KEY, model_name='text-embedding-3-small'),  # API, OpenAI, 1536 dimensions
    embedding_functions.ONNXMiniLM_L6_V2(preferred_providers=['DmlExecutionProvider']),  # Local (GPU), Open Source, 384 dimensions
    embedding_functions.SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2'),  # Local, Open Source, 384 dimensions
]

# Get or create vectorstore
vectorstore_name = 'felipe-dev-picpay-prj-ai-rag-llm-table'
delete_vectorstore(vectorstore_name, database_client)
vectorstore = get_vectorstore(
    name=vectorstore_name,
    embedding_function=embedding_function_callables[0],
    database_client=database_client,
    create=True,
    dimension_count=1536  # 384 for all-MiniLM-L6-v2
)

# Add documents
get_embedding_cost(documents=documents, model_name='text-embedding-3-small')  # If using OpenAI Embeddings
add_documents(vectorstore, documents)  # Add only 10 documents for testing purposes
