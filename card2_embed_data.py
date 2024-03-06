import os
from env import PINECONE_API_KEY, OPENAI_API_KEY
from datetime import date
from typing import Optional, Any, Callable, List, Dict
import json

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

# Get database client
database_client = get_database_client(api_key=PINECONE_API_KEY)

# Get embeddings client
embedding_model_name = 'text-embedding-3-small'
embedding_function = get_embeddings_client(model_name=embedding_model_name, type='api', api_key=OPENAI_API_KEY)


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
    log(vectorstore._index.describe_index_stats())


# Storytelling for each transaction, chunked by 1000 characters
def test_1():
    # Load data
    df = CardTransactions.read()

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

    # Insert documents into vectorstore
    vectorstore_name = 'felipe-dev-picpay-prj-ai-rag-llm-table-1'
    documents_test_1 = redistribute_by_characters(documents, chunk_size=1000, chunk_overlap=50)
    insert_documents(vectorstore_name, documents_test_1)


# Storytelling for each transaction, chunked by transaction
def test_2():
    # Load data
    df = CardTransactions.read()

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

    # Insert documents into vectorstore
    vectorstore_name = 'felipe-dev-picpay-prj-ai-rag-llm-table-2'
    documents_test_2 = documents
    insert_documents(vectorstore_name, documents_test_2)


# Storytelling for each unit in each level of aggregations, mixed chunks scopes
def test_3():
    # Load data
    source_df = CardTransactions.read()
    # source_df = CardTransactions.generate_dummy_data(
    #     order_by=[
    #         'transaction_year',
    #         'portfolio_id',
    #         'consumer_id',
    #         'transaction_at',
    #     ],
    #     n_officers=1,
    #     n_consumers_officer=10,
    #     n_transactions_consumer_day=3,
    #     start_date=date(2023, 1, 1),
    #     end_date=date(2023, 12, 31),
    #     chaos_consumers_officer=0,
    #     chaos_transactions_client_day=0.66,
    #     log=log
    # )

    source_df['product'] = source_df['product'].map(threat_product)
    source_df['card_variant'] = source_df['card_variant'].map(threat_card_variant)

    ## By year, month, day, portfolio, officer, consumer, product, variant, seller, and transaction
    df = source_df.copy()
    level_documents = generate_documents(
        df=df,
        where=None,
        group_by=None,
        order_by=['transaction_year', 'transaction_month', 'transaction_day', 'portfolio_id', 'consumer_id', 'transaction_at'],
        limit=None,
        parse_content_header=lambda record:
            f'''O cliente {record['consumer_name']} (CPF: {record['consumer_document']}), '''
            f'''efetuou um transação de R$ {record['transaction_value']:.2f} '''
            f'''no dia {record['transaction_day']} do mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_day']:02}/{record['transaction_month']:02}/{record['transaction_year']:04}) '''
            f'''com cartão de {record['product']} {record['card_variant']} para o estabelecimento "{record['seller_description']}"''',
        parse_content_body=None,
        parse_metadata=lambda record: dict(record)
    )
    documents = level_documents

    aggregations = {
        'transaction_value_sum': ('transaction_value', 'sum'),
        'transaction_value_max': ('transaction_value', 'max'),
        'transaction_value_min': ('transaction_value', 'min'),
        'transaction_value_count': ('transaction_value', 'count'),
        'transaction_value_mean': ('transaction_value', 'mean'),
        'card_variant_black_count': ('card_variant', lambda x: x.value_counts().loc['BLACK']),
        'card_variant_gold_count': ('card_variant', lambda x: x.value_counts().loc['GOLD']),
        'card_variant_platinum_count': ('card_variant', lambda x: x.value_counts().loc['PLATINUM']),
        'card_variant_standard_count': ('card_variant', lambda x: x.value_counts().loc['STANDARD']),
        'card_variant_international_count': ('card_variant', lambda x: x.value_counts().loc['INTERNACIONAL'])
    }

    ## By year, month, day, consumer, product
    groupby = ['transaction_year', 'transaction_month', 'transaction_day', 'consumer_id', 'consumer_document', 'consumer_name', 'product']
    df = source_df.groupby(groupby, observed=True).agg(**aggregations).reset_index()
    df = df[df.transaction_value_count > 0]
    level_documents = generate_documents(
        df=df,
        where=None,
        group_by=None,
        order_by=groupby,
        limit=None,
        parse_content_header=lambda record:
            f'''Sumário diário de transações do cliente {record['consumer_name']} (CPF: {record['consumer_document']}) com cartão de {record['product']} '''
            f'''para o dia {record['transaction_day']} do mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_day']:02}/{record['transaction_month']:02}/{record['transaction_year']:04}): '''
            f'''- Contagem de transações: {int(record['transaction_value_count'])}, ({int(record['card_variant_black_count'])} com cartão BLACK, '''
            f'''{int(record['card_variant_gold_count'])} com cartão GOLD, {int(record['card_variant_platinum_count'])} com cartão PLATINUM, '''
            f'''{int(record['card_variant_standard_count'])} com cartão STANDARD, {int(record['card_variant_international_count'])} com cartão INTERNACIONAL); '''
            f'''- Valor total: R$ {record['transaction_value_sum']:.2f}; '''
            f'''- Valor médio: R$ {record['transaction_value_mean']:.2f}; '''
            f'''- Valor da maior transação: R$ {record['transaction_value_max']:.2f}; '''
            f'''- Valor da menor transação: R$ {record['transaction_value_min']:.2f}. '''
    )
    documents += level_documents

    ## By year, month, consumer, product
    groupby = ['transaction_year', 'transaction_month', 'consumer_id', 'consumer_document', 'consumer_name', 'product']
    df = source_df.groupby(groupby, observed=True).agg(**aggregations).reset_index()
    df = df[df.transaction_value_count > 0]
    level_documents = generate_documents(
        df=df,
        where=None,
        group_by=None,
        order_by=groupby,
        limit=None,
        parse_content_header=lambda record:
            f'''Sumário mensal de transações do cliente {record['consumer_name']} (CPF: {record['consumer_document']}) com cartão de {record['product']} '''
            f'''para o mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_month']:02}/{record['transaction_year']:04}): '''
            f'''- Contagem de transações: {int(record['transaction_value_count'])}, ({int(record['card_variant_black_count'])} com cartão BLACK, '''
            f'''{int(record['card_variant_gold_count'])} com cartão GOLD, {int(record['card_variant_platinum_count'])} com cartão PLATINUM, '''
            f'''{int(record['card_variant_standard_count'])} com cartão STANDARD, {int(record['card_variant_international_count'])} com cartão INTERNACIONAL); '''
            f'''- Valor total: R$ {record['transaction_value_sum']:.2f}; '''
            f'''- Valor médio: R$ {record['transaction_value_mean']:.2f}; '''
            f'''- Valor da maior transação: R$ {record['transaction_value_max']:.2f}; '''
            f'''- Valor da menor transação: R$ {record['transaction_value_min']:.2f}. '''
    )
    documents += level_documents

    ## By year, consumer, product
    groupby = ['transaction_year', 'consumer_id', 'consumer_document', 'consumer_name', 'product']
    df = source_df.groupby(groupby, observed=True).agg(**aggregations).reset_index()
    df = df[df.transaction_value_count > 0]
    level_documents = generate_documents(
        df=df,
        where=None,
        group_by=None,
        order_by=groupby,
        limit=None,
        parse_content_header=lambda record:
            f'''Sumário anual de transações do cliente {record['consumer_name']} (CPF: {record['consumer_document']}) com cartão de {record['product']} '''
            f'''para o ano de {record['transaction_year']}: '''
            f'''- Contagem de transações: {int(record['transaction_value_count'])}, ({int(record['card_variant_black_count'])} com cartão BLACK, '''
            f'''{int(record['card_variant_gold_count'])} com cartão GOLD, {int(record['card_variant_platinum_count'])} com cartão PLATINUM, '''
            f'''{int(record['card_variant_standard_count'])} com cartão STANDARD, {int(record['card_variant_international_count'])} com cartão INTERNACIONAL); '''
            f'''- Valor total: R$ {record['transaction_value_sum']:.2f}; '''
            f'''- Valor médio: R$ {record['transaction_value_mean']:.2f}; '''
            f'''- Valor da maior transação: R$ {record['transaction_value_max']:.2f}; '''
            f'''- Valor da menor transação: R$ {record['transaction_value_min']:.2f}. '''
    )
    documents += level_documents


    ## By year, month, day, portfolio
    groupby = ['transaction_year', 'transaction_month', 'transaction_day', 'portfolio_id']
    df = source_df.groupby(groupby, observed=True).agg(**aggregations).reset_index()
    df = df[df.transaction_value_count > 0]
    level_documents = generate_documents(
        df=df,
        where=None,
        group_by=None,
        order_by=groupby,
        limit=None,
        parse_content_header=lambda record:
            f'''Sumário diário de transações de todos os clientes da carteira {record['portfolio_id']} '''
            f'''para o dia {record['transaction_day']} do mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_day']:02}/{record['transaction_month']:02}/{record['transaction_year']:04}): '''
            f'''- Contagem de transações: {int(record['transaction_value_count'])}, ({int(record['card_variant_black_count'])} com cartão BLACK, '''
            f'''{int(record['card_variant_gold_count'])} com cartão GOLD, {int(record['card_variant_platinum_count'])} com cartão PLATINUM, '''
            f'''{int(record['card_variant_standard_count'])} com cartão STANDARD, {int(record['card_variant_international_count'])} com cartão INTERNACIONAL); '''
            f'''- Valor total: R$ {record['transaction_value_sum']:.2f}; '''
            f'''- Valor médio: R$ {record['transaction_value_mean']:.2f}; '''
            f'''- Valor da maior transação: R$ {record['transaction_value_max']:.2f}; '''
            f'''- Valor da menor transação: R$ {record['transaction_value_min']:.2f}. '''
    )
    documents += level_documents

    ## By year, month, portfolio
    groupby = ['transaction_year', 'transaction_month', 'portfolio_id']
    df = source_df.groupby(groupby, observed=True).agg(**aggregations).reset_index()
    df = df[df.transaction_value_count > 0]
    level_documents = generate_documents(
        df=df,
        where=None,
        group_by=None,
        order_by=groupby,
        limit=None,
        parse_content_header=lambda record:
            f'''Sumário mensal de transações de todos os clientes da carteira {record['portfolio_id']} '''
            f'''para o mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_month']:02}/{record['transaction_year']:04}): '''
            f'''- Contagem de transações: {int(record['transaction_value_count'])}, ({int(record['card_variant_black_count'])} com cartão BLACK, '''
            f'''{int(record['card_variant_gold_count'])} com cartão GOLD, {int(record['card_variant_platinum_count'])} com cartão PLATINUM, '''
            f'''{int(record['card_variant_standard_count'])} com cartão STANDARD, {int(record['card_variant_international_count'])} com cartão INTERNACIONAL); '''
            f'''- Valor total: R$ {record['transaction_value_sum']:.2f}; '''
            f'''- Valor médio: R$ {record['transaction_value_mean']:.2f}; '''
            f'''- Valor da maior transação: R$ {record['transaction_value_max']:.2f}; '''
            f'''- Valor da menor transação: R$ {record['transaction_value_min']:.2f}. '''
    )
    documents += level_documents

    ## By year, portfolio
    groupby = ['transaction_year', 'portfolio_id']
    df = source_df.groupby(groupby, observed=True).agg(**aggregations).reset_index()
    df = df[df.transaction_value_count > 0]
    level_documents = generate_documents(
        df=df,
        where=None,
        group_by=None,
        order_by=groupby,
        limit=None,
        parse_content_header=lambda record:
            f'''Sumário anual de transações de todos os clientes da carteira {record['portfolio_id']} '''
            f'''para o ano de {record['transaction_year']}: '''
            f'''- Contagem de transações: {int(record['transaction_value_count'])}, ({int(record['card_variant_black_count'])} com cartão BLACK, '''
            f'''{int(record['card_variant_gold_count'])} com cartão GOLD, {int(record['card_variant_platinum_count'])} com cartão PLATINUM, '''
            f'''{int(record['card_variant_standard_count'])} com cartão STANDARD, {int(record['card_variant_international_count'])} com cartão INTERNACIONAL); '''
            f'''- Valor total: R$ {record['transaction_value_sum']:.2f}; '''
            f'''- Valor médio: R$ {record['transaction_value_mean']:.2f}; '''
            f'''- Valor da maior transação: R$ {record['transaction_value_max']:.2f}; '''
            f'''- Valor da menor transação: R$ {record['transaction_value_min']:.2f}. '''
    )
    documents += level_documents

    # Insert documents into vectorstore
    vectorstore_name = 'felipe-dev-picpay-prj-ai-rag-llm-table-3'
    text_path = 'data/refined/card_transactions_documents_test3'
    os.makedirs(text_path, exist_ok=True)
    with open(f'{text_path}/data.txt', 'w') as fo:
        fo.write('\n\n'.join([doc.page_content for doc in documents]))
    insert_documents(vectorstore_name, documents=documents)


# test_1()
# test_2()
test_3()
