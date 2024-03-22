import os
from env import PINECONE_API_KEY, OPENAI_API_KEY
from typing import Any, List, Dict, Tuple

from utils import log, get_month_name
from data_handler import generate_documents, redistribute_by_characters, get_embeddings_client, get_embedding_cost
from data_tables import CardTransactions
from connections.pinecone import (
    get_database_client,
    get_vectorstore,
    delete_vectorstore,
    add_documents,
)

from pandas import DataFrame, concat
from langchain_core.documents import Document


# Configure Logger
log.verbose = True
log.end = '\n\n'

# Get database client
database_client = get_database_client(api_key=PINECONE_API_KEY)
database_type = 'pinecone'

# Get embeddings client
embeddings = {
    'text-embedding-3-small': {
        'dimension_count': 1536,
        'function': get_embeddings_client(model_name='text-embedding-3-small', type='api', api_key=OPENAI_API_KEY)
    },
    'text-embedding-3-large': {
        'dimension_count': 3072,
        'function': get_embeddings_client(model_name='text-embedding-3-large', type='api', api_key=OPENAI_API_KEY)
    },
}


def insert_documents(vectorstore_name: str, documents: List[Document], embedding_model_name: str) -> None:
    embedding_function = embeddings[embedding_model_name]['function']
    embedding_dimension_count = embeddings[embedding_model_name]['dimension_count']

    # Get or create vectorstore
    delete_vectorstore(vectorstore_name, database_client)
    vectorstore = get_vectorstore(
        name=vectorstore_name,
        embedding_function=embedding_function,
        database_client=database_client,
        create=True,
        dimension_count=embedding_dimension_count,
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


def write_documents_txt(vectorstore_name: str, documents: List[Document]) -> None:
    text_path = os.path.join('data', 'refined', database_type, vectorstore_name)
    os.makedirs(text_path, exist_ok=True)
    with open(os.path.join(text_path, 'data.txt'), 'w', encoding='utf-8') as fo:
        fo.write('\n\n'.join([doc.page_content for doc in documents]))


# Standard processing
def test_1(df: DataFrame, aggregations: Dict[str, Tuple[str, Any]], insert: bool = False) -> Tuple[DataFrame, List[Document]]:
    result_dfs = []
    documents = []

    result_dfs.append(CardTransactions.group_by_transaction(df))
    documents = generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''O cliente {record['consumer_name']} (CPF: {record['consumer_document']}) '''
            f'''efetuou a transação de R$ {record['transaction_value']:.2f} '''
            f'''no dia {record['transaction_day']} do mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_day']:02}/{record['transaction_month']:02}/{record['transaction_year']:04}) '''
            f'''com cartão de {record['product']} {record['card_variant']} para o estabelecimento "{record['seller_description']}"'''
    )

    result_dfs.append(CardTransactions.group_by_year_month_day_consumer_product(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Sumário diário de transações do cliente {record['consumer_name']} (CPF: {record['consumer_document']}) com cartão de {record['product']} '''
            f'''no dia {record['transaction_day']} do mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_day']:02}/{record['transaction_month']:02}/{record['transaction_year']:04}): '''
            f'''- Contagem de transações: {int(record['transaction_value_count'])}, ({int(record['card_variant_black_count'])} com cartão BLACK, '''
            f'''{int(record['card_variant_gold_count'])} com cartão GOLD, {int(record['card_variant_platinum_count'])} com cartão PLATINUM, '''
            f'''{int(record['card_variant_standard_count'])} com cartão STANDARD, {int(record['card_variant_international_count'])} com cartão INTERNACIONAL); '''
            f'''- Valor total: R$ {record['transaction_value_sum']:.2f}; '''
            f'''- Valor médio: R$ {record['transaction_value_mean']:.2f}; '''
            f'''- Valor da maior transação: R$ {record['transaction_value_max']:.2f}; '''
            f'''- Valor da menor transação: R$ {record['transaction_value_min']:.2f}. '''
    )

    result_dfs.append(CardTransactions.group_by_year_month_consumer_product(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
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

    result_dfs.append(CardTransactions.group_by_year_consumer_product(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
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
    
    result_dfs.append(CardTransactions.group_by_year_month_day_portfolio(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Sumário diário de transações de todos os clientes da carteira {record['portfolio_id']} '''
            f'''no dia {record['transaction_day']} do mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_day']:02}/{record['transaction_month']:02}/{record['transaction_year']:04}): '''
            f'''- Contagem de transações: {int(record['transaction_value_count'])}, ({int(record['card_variant_black_count'])} com cartão BLACK, '''
            f'''{int(record['card_variant_gold_count'])} com cartão GOLD, {int(record['card_variant_platinum_count'])} com cartão PLATINUM, '''
            f'''{int(record['card_variant_standard_count'])} com cartão STANDARD, {int(record['card_variant_international_count'])} com cartão INTERNACIONAL); '''
            f'''- Valor total: R$ {record['transaction_value_sum']:.2f}; '''
            f'''- Valor médio: R$ {record['transaction_value_mean']:.2f}; '''
            f'''- Valor da maior transação: R$ {record['transaction_value_max']:.2f}; '''
            f'''- Valor da menor transação: R$ {record['transaction_value_min']:.2f}. '''
    )

    result_dfs.append(CardTransactions.group_by_year_month_portfolio(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
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

    result_dfs.append(CardTransactions.group_by_year_portfolio(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
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

    df = concat(result_dfs)
    vectorstore_name = 'prj-ai-rag-llm-table-1-standard'
    embedding_model_name = 'text-embedding-3-small'
    if insert:
        write_documents_txt(vectorstore_name, documents)
        insert_documents(vectorstore_name, documents, embedding_model_name)
    return df, documents


# Discursive text content
def test_2(df: DataFrame, aggregations: Dict[str, Tuple[str, Any]], insert: bool = False) -> Tuple[DataFrame, List[Document]]:
    result_dfs = []
    documents = []

    result_dfs.append(CardTransactions.group_by_transaction(df))
    documents = generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''O cliente {record['consumer_name']} (CPF: {record['consumer_document']}) '''
            f'''efetuou a transação com cartão de {record['product']} {record['card_variant']} de R$ {record['transaction_value']:.2f} '''
            f'''no dia {record['transaction_day']} do mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_day']:02}/{record['transaction_month']:02}/{record['transaction_year']:04}) '''
            f'''para o estabelecimento "{record['seller_description']}"'''
    )

    result_dfs.append(CardTransactions.group_by_year_month_day_consumer_product(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Sumário diário de transações do cliente {record['consumer_name']} (CPF: {record['consumer_document']}) com cartão de {record['product']} '''
            f'''no dia {record['transaction_day']} do mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_day']:02}/{record['transaction_month']:02}/{record['transaction_year']:04}) com um total de '''
            f'''{int(record['transaction_value_count'])} transações, dentre elas {int(record['card_variant_black_count'])} foram realizadas com cartão BLACK, '''
            f'''{int(record['card_variant_gold_count'])} com cartão GOLD, {int(record['card_variant_platinum_count'])} com cartão PLATINUM, '''
            f'''{int(record['card_variant_standard_count'])} com cartão STANDARD e {int(record['card_variant_international_count'])} com cartão INTERNACIONAL, '''
            f'''a soma do valor de todas as transações é de R$ {record['transaction_value_sum']:.2f}, '''
            f'''a média é de R$ {record['transaction_value_mean']:.2f}, '''
            f'''o valor da maior das transações é de R$ {record['transaction_value_max']:.2f} e '''
            f'''o valor da menor das transações é de R$ {record['transaction_value_min']:.2f}. '''
    )

    result_dfs.append(CardTransactions.group_by_year_month_consumer_product(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Sumário mensal de transações do cliente {record['consumer_name']} (CPF: {record['consumer_document']}) com cartão de {record['product']} '''
            f'''para o mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_month']:02}/{record['transaction_year']:04}) com um total de '''
            f'''{int(record['transaction_value_count'])} transações, dentre elas {int(record['card_variant_black_count'])} foram realizadas com cartão BLACK, '''
            f'''{int(record['card_variant_gold_count'])} com cartão GOLD, {int(record['card_variant_platinum_count'])} com cartão PLATINUM, '''
            f'''{int(record['card_variant_standard_count'])} com cartão STANDARD e {int(record['card_variant_international_count'])} com cartão INTERNACIONAL, '''
            f'''a soma do valor de todas as transações é de R$ {record['transaction_value_sum']:.2f}, '''
            f'''a média é de R$ {record['transaction_value_mean']:.2f}, '''
            f'''o valor da maior das transações é de R$ {record['transaction_value_max']:.2f} e '''
            f'''o valor da menor das transações é de R$ {record['transaction_value_min']:.2f}. '''
    )

    result_dfs.append(CardTransactions.group_by_year_consumer_product(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Sumário anual de transações do cliente {record['consumer_name']} (CPF: {record['consumer_document']}) com cartão de {record['product']} '''
            f'''para o ano de {record['transaction_year']} com um total de '''
            f'''{int(record['transaction_value_count'])} transações, dentre elas {int(record['card_variant_black_count'])} foram realizadas com cartão BLACK, '''
            f'''{int(record['card_variant_gold_count'])} com cartão GOLD, {int(record['card_variant_platinum_count'])} com cartão PLATINUM, '''
            f'''{int(record['card_variant_standard_count'])} com cartão STANDARD e {int(record['card_variant_international_count'])} com cartão INTERNACIONAL, '''
            f'''a soma do valor de todas as transações é de R$ {record['transaction_value_sum']:.2f}, '''
            f'''a média é de R$ {record['transaction_value_mean']:.2f}, '''
            f'''o valor da maior das transações é de R$ {record['transaction_value_max']:.2f} e '''
            f'''o valor da menor das transações é de R$ {record['transaction_value_min']:.2f}. '''
    )
    
    result_dfs.append(CardTransactions.group_by_year_month_day_portfolio(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Sumário diário de transações de todos os clientes da carteira {record['portfolio_id']} '''
            f'''no dia {record['transaction_day']} do mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_day']:02}/{record['transaction_month']:02}/{record['transaction_year']:04}) com um total de '''
            f'''{int(record['transaction_value_count'])} transações, dentre elas {int(record['card_variant_black_count'])} foram realizadas com cartão BLACK, '''
            f'''{int(record['card_variant_gold_count'])} com cartão GOLD, {int(record['card_variant_platinum_count'])} com cartão PLATINUM, '''
            f'''{int(record['card_variant_standard_count'])} com cartão STANDARD e {int(record['card_variant_international_count'])} com cartão INTERNACIONAL, '''
            f'''a soma do valor de todas as transações é de R$ {record['transaction_value_sum']:.2f}, '''
            f'''a média é de R$ {record['transaction_value_mean']:.2f}, '''
            f'''o valor da maior das transações é de R$ {record['transaction_value_max']:.2f} e '''
            f'''o valor da menor das transações é de R$ {record['transaction_value_min']:.2f}. '''
    )

    result_dfs.append(CardTransactions.group_by_year_month_portfolio(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Sumário mensal de transações de todos os clientes da carteira {record['portfolio_id']} '''
            f'''para o mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_month']:02}/{record['transaction_year']:04}) com um total de '''
            f'''{int(record['transaction_value_count'])} transações, dentre elas {int(record['card_variant_black_count'])} foram realizadas com cartão BLACK, '''
            f'''{int(record['card_variant_gold_count'])} com cartão GOLD, {int(record['card_variant_platinum_count'])} com cartão PLATINUM, '''
            f'''{int(record['card_variant_standard_count'])} com cartão STANDARD e {int(record['card_variant_international_count'])} com cartão INTERNACIONAL, '''
            f'''a soma do valor de todas as transações é de R$ {record['transaction_value_sum']:.2f}, '''
            f'''a média é de R$ {record['transaction_value_mean']:.2f}, '''
            f'''o valor da maior das transações é de R$ {record['transaction_value_max']:.2f} e '''
            f'''o valor da menor das transações é de R$ {record['transaction_value_min']:.2f}. '''
    )

    result_dfs.append(CardTransactions.group_by_year_portfolio(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Sumário anual de transações de todos os clientes da carteira {record['portfolio_id']} '''
            f'''para o ano de {record['transaction_year']} com um total de '''
            f'''{int(record['transaction_value_count'])} transações, dentre elas {int(record['card_variant_black_count'])} foram realizadas com cartão BLACK, '''
            f'''{int(record['card_variant_gold_count'])} com cartão GOLD, {int(record['card_variant_platinum_count'])} com cartão PLATINUM, '''
            f'''{int(record['card_variant_standard_count'])} com cartão STANDARD e {int(record['card_variant_international_count'])} com cartão INTERNACIONAL, '''
            f'''a soma do valor de todas as transações é de R$ {record['transaction_value_sum']:.2f}, '''
            f'''a média é de R$ {record['transaction_value_mean']:.2f}, '''
            f'''o valor da maior das transações é de R$ {record['transaction_value_max']:.2f} e '''
            f'''o valor da menor das transações é de R$ {record['transaction_value_min']:.2f}. '''
    )

    df = concat(result_dfs)
    vectorstore_name = 'prj-ai-rag-llm-table-2-discursive'
    embedding_model_name = 'text-embedding-3-small'
    if insert:
        write_documents_txt(vectorstore_name, documents)
        insert_documents(vectorstore_name, documents, embedding_model_name)
    return df, documents


# Process credit cards only
def test_3(df: DataFrame, aggregations: Dict[str, Tuple[str, Any]], insert: bool = False) -> Tuple[DataFrame, List[Document]]:
    result_dfs = []
    documents = []

    df = df[df['product'] == 'crédito']

    result_dfs.append(CardTransactions.group_by_transaction(df))
    documents = generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''O cliente {record['consumer_name']} (CPF: {record['consumer_document']}) '''
            f'''efetuou a transação com cartão de crédito {record['card_variant']} no valor de R$ {record['transaction_value']:.2f} '''
            f'''no dia {record['transaction_day']} do mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_day']:02}/{record['transaction_month']:02}/{record['transaction_year']:04}) '''
            f'''para o estabelecimento "{record['seller_description']}"'''
    )

    result_dfs.append(CardTransactions.group_by_year_month_day_consumer(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Sumário diário de transações com cartão de crédito do cliente {record['consumer_name']} (CPF: {record['consumer_document']}) '''
            f'''no dia {record['transaction_day']} do mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_day']:02}/{record['transaction_month']:02}/{record['transaction_year']:04}): '''
            f'''- Contagem de transações: {int(record['transaction_value_count'])}, ({int(record['card_variant_black_count'])} com cartão BLACK, '''
            f'''{int(record['card_variant_gold_count'])} com cartão GOLD, {int(record['card_variant_platinum_count'])} com cartão PLATINUM, '''
            f'''{int(record['card_variant_standard_count'])} com cartão STANDARD, {int(record['card_variant_international_count'])} com cartão INTERNACIONAL); '''
            f'''- Valor total: R$ {record['transaction_value_sum']:.2f}; '''
            f'''- Valor médio: R$ {record['transaction_value_mean']:.2f}; '''
            f'''- Valor da maior transação: R$ {record['transaction_value_max']:.2f}; '''
            f'''- Valor da menor transação: R$ {record['transaction_value_min']:.2f}. '''
    )

    result_dfs.append(CardTransactions.group_by_year_month_consumer(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Sumário mensal de transações com cartão de crédito do cliente {record['consumer_name']} (CPF: {record['consumer_document']}) '''
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

    result_dfs.append(CardTransactions.group_by_year_consumer(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Sumário anual de transações com cartão de crédito do cliente {record['consumer_name']} (CPF: {record['consumer_document']}) '''
            f'''para o ano de {record['transaction_year']}: '''
            f'''- Contagem de transações: {int(record['transaction_value_count'])}, ({int(record['card_variant_black_count'])} com cartão BLACK, '''
            f'''{int(record['card_variant_gold_count'])} com cartão GOLD, {int(record['card_variant_platinum_count'])} com cartão PLATINUM, '''
            f'''{int(record['card_variant_standard_count'])} com cartão STANDARD, {int(record['card_variant_international_count'])} com cartão INTERNACIONAL); '''
            f'''- Valor total: R$ {record['transaction_value_sum']:.2f}; '''
            f'''- Valor médio: R$ {record['transaction_value_mean']:.2f}; '''
            f'''- Valor da maior transação: R$ {record['transaction_value_max']:.2f}; '''
            f'''- Valor da menor transação: R$ {record['transaction_value_min']:.2f}. '''
    )
    
    result_dfs.append(CardTransactions.group_by_year_month_day_portfolio(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Sumário diário de transações com cartão de crédito de todos os clientes da carteira {record['portfolio_id']} '''
            f'''no dia {record['transaction_day']} do mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_day']:02}/{record['transaction_month']:02}/{record['transaction_year']:04}): '''
            f'''- Contagem de transações: {int(record['transaction_value_count'])}, ({int(record['card_variant_black_count'])} com cartão BLACK, '''
            f'''{int(record['card_variant_gold_count'])} com cartão GOLD, {int(record['card_variant_platinum_count'])} com cartão PLATINUM, '''
            f'''{int(record['card_variant_standard_count'])} com cartão STANDARD, {int(record['card_variant_international_count'])} com cartão INTERNACIONAL); '''
            f'''- Valor total: R$ {record['transaction_value_sum']:.2f}; '''
            f'''- Valor médio: R$ {record['transaction_value_mean']:.2f}; '''
            f'''- Valor da maior transação: R$ {record['transaction_value_max']:.2f}; '''
            f'''- Valor da menor transação: R$ {record['transaction_value_min']:.2f}. '''
    )

    result_dfs.append(CardTransactions.group_by_year_month_portfolio(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Sumário mensal de transações com cartão de crédito de todos os clientes da carteira {record['portfolio_id']} '''
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

    result_dfs.append(CardTransactions.group_by_year_portfolio(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Sumário anual de transações com cartão de crédito de todos os clientes da carteira {record['portfolio_id']} '''
            f'''para o ano de {record['transaction_year']}: '''
            f'''- Contagem de transações: {int(record['transaction_value_count'])}, ({int(record['card_variant_black_count'])} com cartão BLACK, '''
            f'''{int(record['card_variant_gold_count'])} com cartão GOLD, {int(record['card_variant_platinum_count'])} com cartão PLATINUM, '''
            f'''{int(record['card_variant_standard_count'])} com cartão STANDARD, {int(record['card_variant_international_count'])} com cartão INTERNACIONAL); '''
            f'''- Valor total: R$ {record['transaction_value_sum']:.2f}; '''
            f'''- Valor médio: R$ {record['transaction_value_mean']:.2f}; '''
            f'''- Valor da maior transação: R$ {record['transaction_value_max']:.2f}; '''
            f'''- Valor da menor transação: R$ {record['transaction_value_min']:.2f}. '''
    )

    df = concat(result_dfs)
    vectorstore_name = 'prj-ai-rag-llm-table-3-standard-creditcard'
    embedding_model_name = 'text-embedding-3-small'
    if insert:
        write_documents_txt(vectorstore_name, documents)
        insert_documents(vectorstore_name, documents, embedding_model_name)
    return df, documents


# Discursive text content and only credit cards data
def test_4(df: DataFrame, aggregations: Dict[str, Tuple[str, Any]], insert: bool = False) -> Tuple[DataFrame, List[Document]]:
    result_dfs = []
    documents = []

    df = df[df['product'] == 'crédito']

    result_dfs.append(CardTransactions.group_by_transaction(df))
    documents = generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''O cliente {record['consumer_name']} (CPF: {record['consumer_document']}) '''
            f'''efetuou a transação com cartão de crédito {record['card_variant']} de R$ {record['transaction_value']:.2f} '''
            f'''no dia {record['transaction_day']} do mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_day']:02}/{record['transaction_month']:02}/{record['transaction_year']:04}) '''
            f'''para o estabelecimento "{record['seller_description']}"'''
    )

    result_dfs.append(CardTransactions.group_by_year_month_day_consumer(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Sumário diário de transações com cartão de crédito do cliente {record['consumer_name']} (CPF: {record['consumer_document']}) '''
            f'''no dia {record['transaction_day']} do mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_day']:02}/{record['transaction_month']:02}/{record['transaction_year']:04}) com um total de '''
            f'''{int(record['transaction_value_count'])} transações, dentre elas {int(record['card_variant_black_count'])} foram realizadas com cartão BLACK, '''
            f'''{int(record['card_variant_gold_count'])} com cartão GOLD, {int(record['card_variant_platinum_count'])} com cartão PLATINUM, '''
            f'''{int(record['card_variant_standard_count'])} com cartão STANDARD e {int(record['card_variant_international_count'])} com cartão INTERNACIONAL, '''
            f'''a soma do valor de todas as transações é de R$ {record['transaction_value_sum']:.2f}, '''
            f'''a média é de R$ {record['transaction_value_mean']:.2f}, '''
            f'''o valor da maior das transações é de R$ {record['transaction_value_max']:.2f} e '''
            f'''o valor da menor das transações é de R$ {record['transaction_value_min']:.2f}. '''
    )

    result_dfs.append(CardTransactions.group_by_year_month_consumer(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Sumário mensal de transações com cartão de crédito do cliente {record['consumer_name']} (CPF: {record['consumer_document']}) '''
            f'''para o mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_month']:02}/{record['transaction_year']:04}) com um total de '''
            f'''{int(record['transaction_value_count'])} transações, dentre elas {int(record['card_variant_black_count'])} foram realizadas com cartão BLACK, '''
            f'''{int(record['card_variant_gold_count'])} com cartão GOLD, {int(record['card_variant_platinum_count'])} com cartão PLATINUM, '''
            f'''{int(record['card_variant_standard_count'])} com cartão STANDARD e {int(record['card_variant_international_count'])} com cartão INTERNACIONAL, '''
            f'''a soma do valor de todas as transações é de R$ {record['transaction_value_sum']:.2f}, '''
            f'''a média é de R$ {record['transaction_value_mean']:.2f}, '''
            f'''o valor da maior das transações é de R$ {record['transaction_value_max']:.2f} e '''
            f'''o valor da menor das transações é de R$ {record['transaction_value_min']:.2f}. '''
    )

    result_dfs.append(CardTransactions.group_by_year_consumer(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Sumário anual de transações com cartão de crédito do cliente {record['consumer_name']} (CPF: {record['consumer_document']}) '''
            f'''para o ano de {record['transaction_year']} com um total de '''
            f'''{int(record['transaction_value_count'])} transações, dentre elas {int(record['card_variant_black_count'])} foram realizadas com cartão BLACK, '''
            f'''{int(record['card_variant_gold_count'])} com cartão GOLD, {int(record['card_variant_platinum_count'])} com cartão PLATINUM, '''
            f'''{int(record['card_variant_standard_count'])} com cartão STANDARD e {int(record['card_variant_international_count'])} com cartão INTERNACIONAL, '''
            f'''a soma do valor de todas as transações é de R$ {record['transaction_value_sum']:.2f}, '''
            f'''a média é de R$ {record['transaction_value_mean']:.2f}, '''
            f'''o valor da maior das transações é de R$ {record['transaction_value_max']:.2f} e '''
            f'''o valor da menor das transações é de R$ {record['transaction_value_min']:.2f}. '''
    )
    
    result_dfs.append(CardTransactions.group_by_year_month_day_portfolio(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Sumário diário de transações com cartão de crédito de todos os clientes da carteira {record['portfolio_id']} '''
            f'''no dia {record['transaction_day']} do mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_day']:02}/{record['transaction_month']:02}/{record['transaction_year']:04}) com um total de '''
            f'''{int(record['transaction_value_count'])} transações, dentre elas {int(record['card_variant_black_count'])} foram realizadas com cartão BLACK, '''
            f'''{int(record['card_variant_gold_count'])} com cartão GOLD, {int(record['card_variant_platinum_count'])} com cartão PLATINUM, '''
            f'''{int(record['card_variant_standard_count'])} com cartão STANDARD e {int(record['card_variant_international_count'])} com cartão INTERNACIONAL, '''
            f'''a soma do valor de todas as transações é de R$ {record['transaction_value_sum']:.2f}, '''
            f'''a média é de R$ {record['transaction_value_mean']:.2f}, '''
            f'''o valor da maior das transações é de R$ {record['transaction_value_max']:.2f} e '''
            f'''o valor da menor das transações é de R$ {record['transaction_value_min']:.2f}. '''
    )

    result_dfs.append(CardTransactions.group_by_year_month_portfolio(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Sumário mensal de transações com cartão de crédito de todos os clientes da carteira {record['portfolio_id']} '''
            f'''para o mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_month']:02}/{record['transaction_year']:04}) com um total de '''
            f'''{int(record['transaction_value_count'])} transações, dentre elas {int(record['card_variant_black_count'])} foram realizadas com cartão BLACK, '''
            f'''{int(record['card_variant_gold_count'])} com cartão GOLD, {int(record['card_variant_platinum_count'])} com cartão PLATINUM, '''
            f'''{int(record['card_variant_standard_count'])} com cartão STANDARD e {int(record['card_variant_international_count'])} com cartão INTERNACIONAL, '''
            f'''a soma do valor de todas as transações é de R$ {record['transaction_value_sum']:.2f}, '''
            f'''a média é de R$ {record['transaction_value_mean']:.2f}, '''
            f'''o valor da maior das transações é de R$ {record['transaction_value_max']:.2f} e '''
            f'''o valor da menor das transações é de R$ {record['transaction_value_min']:.2f}. '''
    )

    result_dfs.append(CardTransactions.group_by_year_portfolio(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Sumário anual de transações com cartão de crédito de todos os clientes da carteira {record['portfolio_id']} '''
            f'''para o ano de {record['transaction_year']} com um total de '''
            f'''{int(record['transaction_value_count'])} transações, dentre elas {int(record['card_variant_black_count'])} foram realizadas com cartão BLACK, '''
            f'''{int(record['card_variant_gold_count'])} com cartão GOLD, {int(record['card_variant_platinum_count'])} com cartão PLATINUM, '''
            f'''{int(record['card_variant_standard_count'])} com cartão STANDARD e {int(record['card_variant_international_count'])} com cartão INTERNACIONAL, '''
            f'''a soma do valor de todas as transações é de R$ {record['transaction_value_sum']:.2f}, '''
            f'''a média é de R$ {record['transaction_value_mean']:.2f}, '''
            f'''o valor da maior das transações é de R$ {record['transaction_value_max']:.2f} e '''
            f'''o valor da menor das transações é de R$ {record['transaction_value_min']:.2f}. '''
    )

    df = concat(result_dfs)
    vectorstore_name = 'prj-ai-rag-llm-table-4-discursive-creditcard'
    embedding_model_name = 'text-embedding-3-small'
    if insert:
        write_documents_txt(vectorstore_name, documents)
        insert_documents(vectorstore_name, documents, embedding_model_name)
    return df, documents


# Join transactions with diary summaries
def test_5(df: DataFrame, aggregations: Dict[str, Tuple[str, Any]], insert: bool = False) -> Tuple[DataFrame, List[Document]]:
    result_dfs = []
    documents = []

    df = df[df['product'] == 'crédito']

    result_dfs.append(CardTransactions.group_by_transaction(df))
    documents += generate_documents(
        df=result_dfs[-1],
        group_by=['transaction_year', 'transaction_month', 'transaction_day', 'consumer_id', 'consumer_document', 'consumer_name', 'product'],
        parse_content_header=lambda record:
            f'''Sumário diário de transações com cartão de crédito do cliente {record['consumer_name']} (CPF: {record['consumer_document']}) '''
            f'''no dia {record['transaction_day']} do mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_day']:02}/{record['transaction_month']:02}/{record['transaction_year']:04}): ''',
        parse_content_body=lambda record:
            f'''\tR$ {record['transaction_value']:.2f} no estabelecimento "{record['seller_description']}";'''
    )

    result_dfs.append(CardTransactions.group_by_year_month_consumer(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Sumário mensal de transações com cartão de crédito do cliente {record['consumer_name']} (CPF: {record['consumer_document']}) '''
            f'''para o mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_month']:02}/{record['transaction_year']:04}) com um total de '''
            f'''{int(record['transaction_value_count'])} transações, dentre elas {int(record['card_variant_black_count'])} foram realizadas com cartão BLACK, '''
            f'''{int(record['card_variant_gold_count'])} com cartão GOLD, {int(record['card_variant_platinum_count'])} com cartão PLATINUM, '''
            f'''{int(record['card_variant_standard_count'])} com cartão STANDARD e {int(record['card_variant_international_count'])} com cartão INTERNACIONAL, '''
            f'''a soma do valor de todas as transações é de R$ {record['transaction_value_sum']:.2f}, '''
            f'''a média é de R$ {record['transaction_value_mean']:.2f}, '''
            f'''o valor da maior das transações é de R$ {record['transaction_value_max']:.2f} e '''
            f'''o valor da menor das transações é de R$ {record['transaction_value_min']:.2f}. '''
    )

    result_dfs.append(CardTransactions.group_by_year_consumer(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Sumário anual de transações com cartão de crédito do cliente {record['consumer_name']} (CPF: {record['consumer_document']}) '''
            f'''para o ano de {record['transaction_year']} com um total de '''
            f'''{int(record['transaction_value_count'])} transações, dentre elas {int(record['card_variant_black_count'])} foram realizadas com cartão BLACK, '''
            f'''{int(record['card_variant_gold_count'])} com cartão GOLD, {int(record['card_variant_platinum_count'])} com cartão PLATINUM, '''
            f'''{int(record['card_variant_standard_count'])} com cartão STANDARD e {int(record['card_variant_international_count'])} com cartão INTERNACIONAL, '''
            f'''a soma do valor de todas as transações é de R$ {record['transaction_value_sum']:.2f}, '''
            f'''a média é de R$ {record['transaction_value_mean']:.2f}, '''
            f'''o valor da maior das transações é de R$ {record['transaction_value_max']:.2f} e '''
            f'''o valor da menor das transações é de R$ {record['transaction_value_min']:.2f}. '''
    )
    
    result_dfs.append(CardTransactions.group_by_year_month_day_portfolio(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Sumário diário de transações com cartão de crédito de todos os clientes da carteira {record['portfolio_id']} '''
            f'''no dia {record['transaction_day']} do mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_day']:02}/{record['transaction_month']:02}/{record['transaction_year']:04}) com um total de '''
            f'''{int(record['transaction_value_count'])} transações, dentre elas {int(record['card_variant_black_count'])} foram realizadas com cartão BLACK, '''
            f'''{int(record['card_variant_gold_count'])} com cartão GOLD, {int(record['card_variant_platinum_count'])} com cartão PLATINUM, '''
            f'''{int(record['card_variant_standard_count'])} com cartão STANDARD e {int(record['card_variant_international_count'])} com cartão INTERNACIONAL, '''
            f'''a soma do valor de todas as transações é de R$ {record['transaction_value_sum']:.2f}, '''
            f'''a média é de R$ {record['transaction_value_mean']:.2f}, '''
            f'''o valor da maior das transações é de R$ {record['transaction_value_max']:.2f} e '''
            f'''o valor da menor das transações é de R$ {record['transaction_value_min']:.2f}. '''
    )

    result_dfs.append(CardTransactions.group_by_year_month_portfolio(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Sumário mensal de transações com cartão de crédito de todos os clientes da carteira {record['portfolio_id']} '''
            f'''para o mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_month']:02}/{record['transaction_year']:04}) com um total de '''
            f'''{int(record['transaction_value_count'])} transações, dentre elas {int(record['card_variant_black_count'])} foram realizadas com cartão BLACK, '''
            f'''{int(record['card_variant_gold_count'])} com cartão GOLD, {int(record['card_variant_platinum_count'])} com cartão PLATINUM, '''
            f'''{int(record['card_variant_standard_count'])} com cartão STANDARD e {int(record['card_variant_international_count'])} com cartão INTERNACIONAL, '''
            f'''a soma do valor de todas as transações é de R$ {record['transaction_value_sum']:.2f}, '''
            f'''a média é de R$ {record['transaction_value_mean']:.2f}, '''
            f'''o valor da maior das transações é de R$ {record['transaction_value_max']:.2f} e '''
            f'''o valor da menor das transações é de R$ {record['transaction_value_min']:.2f}. '''
    )

    result_dfs.append(CardTransactions.group_by_year_portfolio(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Sumário anual de transações com cartão de crédito de todos os clientes da carteira {record['portfolio_id']} '''
            f'''para o ano de {record['transaction_year']} com um total de '''
            f'''{int(record['transaction_value_count'])} transações, dentre elas {int(record['card_variant_black_count'])} foram realizadas com cartão BLACK, '''
            f'''{int(record['card_variant_gold_count'])} com cartão GOLD, {int(record['card_variant_platinum_count'])} com cartão PLATINUM, '''
            f'''{int(record['card_variant_standard_count'])} com cartão STANDARD e {int(record['card_variant_international_count'])} com cartão INTERNACIONAL, '''
            f'''a soma do valor de todas as transações é de R$ {record['transaction_value_sum']:.2f}, '''
            f'''a média é de R$ {record['transaction_value_mean']:.2f}, '''
            f'''o valor da maior das transações é de R$ {record['transaction_value_max']:.2f} e '''
            f'''o valor da menor das transações é de R$ {record['transaction_value_min']:.2f}. '''
    )

    df = concat(result_dfs)
    vectorstore_name = 'prj-ai-rag-llm-table-5-join-wo-transactions'
    embedding_model_name = 'text-embedding-3-small'
    if insert:
        write_documents_txt(vectorstore_name, documents)
        insert_documents(vectorstore_name, documents, embedding_model_name)
    return df, documents


# Chunks of 1000 tokens
def test_6(df: DataFrame, aggregations: Dict[str, Tuple[str, Any]], insert: bool = False) -> Tuple[DataFrame, List[Document]]:
    df, documents = test_5(df, aggregations, insert=False)
    documents = redistribute_by_characters(documents, 1000, 0)
    vectorstore_name = 'prj-ai-rag-llm-table-6-chunks'
    embedding_model_name = 'text-embedding-3-small'
    if insert:
        write_documents_txt(vectorstore_name, documents)
        insert_documents(vectorstore_name, documents, embedding_model_name)
    return df, documents


# Cohere embeddings
def test_7(df: DataFrame, aggregations: Dict[str, Tuple[str, Any]], insert: bool = False) -> Tuple[DataFrame, List[Document]]:
    df, documents = test_5(df, aggregations, insert=False)
    documents = redistribute_by_characters(documents, 1000, 0)

    # Cohere
    database_client = CohereClient()
    vectorstore_name = 'prj-ai-rag-llm-table-7-cohere'
    embedding_model_name = 'text-embedding-3-small'

    # Insert documents
    if insert:
        return
    return df, documents


# Day becomes "no dia", Month becomes "resumo", Year becomes "sumário"
def test_8(df: DataFrame, aggregations: Dict[str, Tuple[str, Any]], insert: bool = False) -> Tuple[DataFrame, List[Document]]:
    result_dfs = []
    documents = []

    df = df[df['product'] == 'crédito']

    result_dfs.append(CardTransactions.group_by_transaction(df))
    documents += generate_documents(
        df=result_dfs[-1],
        group_by=['transaction_year', 'transaction_month', 'transaction_day', 'consumer_id', 'consumer_document', 'consumer_name', 'product'],
        parse_content_header=lambda record:
            f'''O cliente {record['consumer_name']} (CPF: {record['consumer_document']}) realizou '''
            f'''no dia {record['transaction_day']} do mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_day']:02}/{record['transaction_month']:02}/{record['transaction_year']:04}) '''
            f'''as seguintes transações com cartão de crédito: ''',
        parse_content_body=lambda record:
            f'''\tR$ {record['transaction_value']:.2f} no estabelecimento "{record['seller_description']}";'''
    )

    result_dfs.append(CardTransactions.group_by_year_month_consumer(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Resumo mensal das transações com cartão de crédito do cliente {record['consumer_name']} (CPF: {record['consumer_document']}) '''
            f'''no mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_month']:02}/{record['transaction_year']:04}) com um total de '''
            f'''{int(record['transaction_value_count'])} transações, dentre elas {int(record['card_variant_black_count'])} foram realizadas com cartão BLACK, '''
            f'''{int(record['card_variant_gold_count'])} com cartão GOLD, {int(record['card_variant_platinum_count'])} com cartão PLATINUM, '''
            f'''{int(record['card_variant_standard_count'])} com cartão STANDARD e {int(record['card_variant_international_count'])} com cartão INTERNACIONAL, '''
            f'''a soma do valor de todas as transações é de R$ {record['transaction_value_sum']:.2f}, '''
            f'''a média é de R$ {record['transaction_value_mean']:.2f}, '''
            f'''o valor da maior das transações é de R$ {record['transaction_value_max']:.2f} e '''
            f'''o valor da menor das transações é de R$ {record['transaction_value_min']:.2f}. '''
    )

    result_dfs.append(CardTransactions.group_by_year_consumer(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Sumário anual de transações com cartão de crédito do cliente {record['consumer_name']} (CPF: {record['consumer_document']}) '''
            f'''no ano de {record['transaction_year']} com um total de '''
            f'''{int(record['transaction_value_count'])} transações, dentre elas {int(record['card_variant_black_count'])} foram realizadas com cartão BLACK, '''
            f'''{int(record['card_variant_gold_count'])} com cartão GOLD, {int(record['card_variant_platinum_count'])} com cartão PLATINUM, '''
            f'''{int(record['card_variant_standard_count'])} com cartão STANDARD e {int(record['card_variant_international_count'])} com cartão INTERNACIONAL, '''
            f'''a soma do valor de todas as transações é de R$ {record['transaction_value_sum']:.2f}, '''
            f'''a média é de R$ {record['transaction_value_mean']:.2f}, '''
            f'''o valor da maior das transações é de R$ {record['transaction_value_max']:.2f} e '''
            f'''o valor da menor das transações é de R$ {record['transaction_value_min']:.2f}. '''
    )
    
    result_dfs.append(CardTransactions.group_by_year_month_day_portfolio(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Sumário diário de transações com cartão de crédito de todos os clientes da carteira {record['portfolio_id']} '''
            f'''no dia {record['transaction_day']} do mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_day']:02}/{record['transaction_month']:02}/{record['transaction_year']:04}) com um total de '''
            f'''{int(record['transaction_value_count'])} transações, dentre elas {int(record['card_variant_black_count'])} foram realizadas com cartão BLACK, '''
            f'''{int(record['card_variant_gold_count'])} com cartão GOLD, {int(record['card_variant_platinum_count'])} com cartão PLATINUM, '''
            f'''{int(record['card_variant_standard_count'])} com cartão STANDARD e {int(record['card_variant_international_count'])} com cartão INTERNACIONAL, '''
            f'''a soma do valor de todas as transações é de R$ {record['transaction_value_sum']:.2f}, '''
            f'''a média é de R$ {record['transaction_value_mean']:.2f}, '''
            f'''o valor da maior das transações é de R$ {record['transaction_value_max']:.2f} e '''
            f'''o valor da menor das transações é de R$ {record['transaction_value_min']:.2f}. '''
    )

    result_dfs.append(CardTransactions.group_by_year_month_portfolio(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Sumário mensal de transações com cartão de crédito de todos os clientes da carteira {record['portfolio_id']} '''
            f'''para o mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_month']:02}/{record['transaction_year']:04}) com um total de '''
            f'''{int(record['transaction_value_count'])} transações, dentre elas {int(record['card_variant_black_count'])} foram realizadas com cartão BLACK, '''
            f'''{int(record['card_variant_gold_count'])} com cartão GOLD, {int(record['card_variant_platinum_count'])} com cartão PLATINUM, '''
            f'''{int(record['card_variant_standard_count'])} com cartão STANDARD e {int(record['card_variant_international_count'])} com cartão INTERNACIONAL, '''
            f'''a soma do valor de todas as transações é de R$ {record['transaction_value_sum']:.2f}, '''
            f'''a média é de R$ {record['transaction_value_mean']:.2f}, '''
            f'''o valor da maior das transações é de R$ {record['transaction_value_max']:.2f} e '''
            f'''o valor da menor das transações é de R$ {record['transaction_value_min']:.2f}. '''
    )

    result_dfs.append(CardTransactions.group_by_year_portfolio(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Sumário anual de transações com cartão de crédito de todos os clientes da carteira {record['portfolio_id']} '''
            f'''para o ano de {record['transaction_year']} com um total de '''
            f'''{int(record['transaction_value_count'])} transações, dentre elas {int(record['card_variant_black_count'])} foram realizadas com cartão BLACK, '''
            f'''{int(record['card_variant_gold_count'])} com cartão GOLD, {int(record['card_variant_platinum_count'])} com cartão PLATINUM, '''
            f'''{int(record['card_variant_standard_count'])} com cartão STANDARD e {int(record['card_variant_international_count'])} com cartão INTERNACIONAL, '''
            f'''a soma do valor de todas as transações é de R$ {record['transaction_value_sum']:.2f}, '''
            f'''a média é de R$ {record['transaction_value_mean']:.2f}, '''
            f'''o valor da maior das transações é de R$ {record['transaction_value_max']:.2f} e '''
            f'''o valor da menor das transações é de R$ {record['transaction_value_min']:.2f}. '''
    )

    df = concat(result_dfs)
    vectorstore_name = 'prj-ai-rag-llm-table-8-storytelling'
    embedding_model_name = 'text-embedding-3-small'
    if insert:
        write_documents_txt(vectorstore_name, documents)
        insert_documents(vectorstore_name, documents, embedding_model_name)
    return df, documents


# Questions included in the document
def test_9(df: DataFrame, aggregations: Dict[str, Tuple[str, Any]], insert: bool = False) -> Tuple[DataFrame, List[Document]]:
    result_dfs = []
    documents = []

    df = df[df['product'] == 'crédito']

    result_dfs.append(CardTransactions.group_by_transaction(df))
    documents += generate_documents(
        df=result_dfs[-1],
        group_by=['transaction_year', 'transaction_month', 'transaction_day', 'consumer_id', 'consumer_document', 'consumer_name', 'product'],
        parse_content_header=lambda record:
            f'''[Quanto o cliente {record['consumer_name']} (CPF: {record['consumer_document']}) transacionou com cartão de crédito '''
            f'''no dia {record['transaction_day']} do mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_day']:02}/{record['transaction_month']:02}/{record['transaction_year']:04})?]\t'''
            f'''Resumo diário das transaçoes do cliente no dia: ''',
        parse_content_body=lambda record:
            f'''\tR$ {record['transaction_value']:.2f} no estabelecimento "{record['seller_description']}";'''
    )

    result_dfs.append(CardTransactions.group_by_year_month_consumer(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''[Quanto o cliente {record['consumer_name']} (CPF: {record['consumer_document']}) transacionou no total com cartão de crédito '''
            f'''no mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''{record['transaction_month']:02}/{record['transaction_year']:04})?]\t'''
            f'''Resumo mensal das transações do cliente no mês, '''
            f'''com uma contagem total de {int(record['transaction_value_count'])} transações '''
            f'''e valor total de R$ {record['transaction_value_sum']:.2f}, '''
            f'''o valor médio é de R$ {record['transaction_value_mean']:.2f}, '''
            f'''o valor da maior transaçao é de R$ {record['transaction_value_max']:.2f} e '''
            f'''o valor da menor transação é de R$ {record['transaction_value_min']:.2f}. '''
            f'''Dentre elas foram realizadas ''' + ', '.join(o for o in [
                f'''{int(record['card_variant_black_count'])} com cartão BLACK''' if int(record['card_variant_black_count']) > 0 else None,
                f'''{int(record['card_variant_gold_count'])} com cartão GOLD''' if int(record['card_variant_gold_count']) > 0 else None,
                f'''{int(record['card_variant_platinum_count'])} com cartão PLATINUM''' if int(record['card_variant_platinum_count']) > 0 else None,
                f'''{int(record['card_variant_standard_count'])} com cartão STANDARD''' if int(record['card_variant_standard_count']) > 0 else None,
                f'''{int(record['card_variant_international_count'])} com cartão INTERNACIONAL''' if int(record['card_variant_international_count']) > 0 else None
            ] if o) + '.'
    )

    result_dfs.append(CardTransactions.group_by_year_consumer(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''[Quanto o cliente {record['consumer_name']} (CPF: {record['consumer_document']}) transacionou no total com cartão de crédito '''
            f'''no ano de {record['transaction_year']}?]\t'''
            f'''Resumo anual de transações com cartão de crédito do cliente no ano, '''
            f'''com uma contagem total de {int(record['transaction_value_count'])} transações '''
            f'''e valor total de R$ {record['transaction_value_sum']:.2f}, '''
            f'''o valor médio é de R$ {record['transaction_value_mean']:.2f}, '''
            f'''o valor da maior transaçao é de R$ {record['transaction_value_max']:.2f} e '''
            f'''o valor da menor transação é de R$ {record['transaction_value_min']:.2f}. '''
            f'''Dentre elas foram realizadas ''' + ', '.join(o for o in [
                f'''{int(record['card_variant_black_count'])} com cartão BLACK''' if int(record['card_variant_black_count']) > 0 else None,
                f'''{int(record['card_variant_gold_count'])} com cartão GOLD''' if int(record['card_variant_gold_count']) > 0 else None,
                f'''{int(record['card_variant_platinum_count'])} com cartão PLATINUM''' if int(record['card_variant_platinum_count']) > 0 else None,
                f'''{int(record['card_variant_standard_count'])} com cartão STANDARD''' if int(record['card_variant_standard_count']) > 0 else None,
                f'''{int(record['card_variant_international_count'])} com cartão INTERNACIONAL''' if int(record['card_variant_international_count']) > 0 else None
            ] if o) + '.'
    )
    
    result_dfs.append(CardTransactions.group_by_year_month_day_portfolio(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''[Qual o total transacionado com cartão de crédito por todos os clientes '''
            f'''no dia {record['transaction_day']} do mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_day']:02}/{record['transaction_month']:02}/{record['transaction_year']:04})?]\t'''
            f'''Resumo diário de todas as transaçoes no dia, '''
            f'''com uma contagem total de {int(record['transaction_value_count'])} transações '''
            f'''e valor total de R$ {record['transaction_value_sum']:.2f}, '''
            f'''o valor médio é de R$ {record['transaction_value_mean']:.2f}, '''
            f'''o valor da maior transaçao é de R$ {record['transaction_value_max']:.2f} e '''
            f'''o valor da menor transação é de R$ {record['transaction_value_min']:.2f}. '''
            f'''Dentre elas foram realizadas ''' + ', '.join(o for o in [
                f'''{int(record['card_variant_black_count'])} com cartão BLACK''' if int(record['card_variant_black_count']) > 0 else None,
                f'''{int(record['card_variant_gold_count'])} com cartão GOLD''' if int(record['card_variant_gold_count']) > 0 else None,
                f'''{int(record['card_variant_platinum_count'])} com cartão PLATINUM''' if int(record['card_variant_platinum_count']) > 0 else None,
                f'''{int(record['card_variant_standard_count'])} com cartão STANDARD''' if int(record['card_variant_standard_count']) > 0 else None,
                f'''{int(record['card_variant_international_count'])} com cartão INTERNACIONAL''' if int(record['card_variant_international_count']) > 0 else None
            ] if o) + '.'
    )

    result_dfs.append(CardTransactions.group_by_year_month_portfolio(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''[Qual o total transacionado com cartão de crédito por todos os clientes '''
            f'''no mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_month']:02}/{record['transaction_year']:04})?]\t'''
            f'''Resumo mensal de todas as transaçoes no mês, '''
            f'''com uma contagem total de {int(record['transaction_value_count'])} transações '''
            f'''e valor total de R$ {record['transaction_value_sum']:.2f}, '''
            f'''o valor médio é de R$ {record['transaction_value_mean']:.2f}, '''
            f'''o valor da maior transaçao é de R$ {record['transaction_value_max']:.2f} e '''
            f'''o valor da menor transação é de R$ {record['transaction_value_min']:.2f}. '''
            f'''Dentre elas foram realizadas ''' + ', '.join(o for o in [
                f'''{int(record['card_variant_black_count'])} com cartão BLACK''' if int(record['card_variant_black_count']) > 0 else None,
                f'''{int(record['card_variant_gold_count'])} com cartão GOLD''' if int(record['card_variant_gold_count']) > 0 else None,
                f'''{int(record['card_variant_platinum_count'])} com cartão PLATINUM''' if int(record['card_variant_platinum_count']) > 0 else None,
                f'''{int(record['card_variant_standard_count'])} com cartão STANDARD''' if int(record['card_variant_standard_count']) > 0 else None,
                f'''{int(record['card_variant_international_count'])} com cartão INTERNACIONAL''' if int(record['card_variant_international_count']) > 0 else None
            ] if o) + '.'
    )

    result_dfs.append(CardTransactions.group_by_year_portfolio(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''[Qual o total transacionado com cartão de crédito por todos os clientes '''
            f'''no ano de {record['transaction_year']}?]\t'''
            f'''Resumo anual de todas as transaçoes no ano, '''
            f'''com uma contagem total de {int(record['transaction_value_count'])} transações '''
            f'''e valor total de R$ {record['transaction_value_sum']:.2f}, '''
            f'''o valor médio é de R$ {record['transaction_value_mean']:.2f}, '''
            f'''o valor da maior transaçao é de R$ {record['transaction_value_max']:.2f} e '''
            f'''o valor da menor transação é de R$ {record['transaction_value_min']:.2f}. '''
            f'''Dentre elas foram realizadas ''' + ', '.join(o for o in [
                f'''{int(record['card_variant_black_count'])} com cartão BLACK''' if int(record['card_variant_black_count']) > 0 else None,
                f'''{int(record['card_variant_gold_count'])} com cartão GOLD''' if int(record['card_variant_gold_count']) > 0 else None,
                f'''{int(record['card_variant_platinum_count'])} com cartão PLATINUM''' if int(record['card_variant_platinum_count']) > 0 else None,
                f'''{int(record['card_variant_standard_count'])} com cartão STANDARD''' if int(record['card_variant_standard_count']) > 0 else None,
                f'''{int(record['card_variant_international_count'])} com cartão INTERNACIONAL''' if int(record['card_variant_international_count']) > 0 else None
            ] if o) + '.'
    )

    df = concat(result_dfs)
    vectorstore_name = 'prj-ai-rag-llm-table-9-questions'
    embedding_model_name = 'text-embedding-3-small'
    if insert:
        write_documents_txt(vectorstore_name, documents)
        insert_documents(vectorstore_name, documents, embedding_model_name)
    return df, documents


# Store granularity in metadata
def test_10(df: DataFrame, aggregations: Dict[str, Tuple[str, Any]], insert: bool = False) -> Tuple[DataFrame, List[Document]]:
    result_dfs = []
    documents = []

    df = df[df['product'] == 'crédito']

    result_dfs.append(CardTransactions.group_by_transaction(df))
    documents = generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''O cliente {record['consumer_name']} (CPF: {record['consumer_document']}) '''
            f'''efetuou a transação de ID "{record['transaction_id']}" com cartão de crédito {record['card_variant']} de R$ {record['transaction_value']:.2f} '''
            f'''no dia {record['transaction_day']} do mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_day']:02}/{record['transaction_month']:02}/{record['transaction_year']:04}) '''
            f'''para o estabelecimento "{record['seller_description']}"'''
    )
    
    result_dfs.append(CardTransactions.group_by_year_month_day_consumer(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Resumo diário das transações com cartão de crédito do cliente {record['consumer_name']} (CPF: {record['consumer_document']}) '''
            f'''no dia {record['transaction_day']} do mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_day']:02}/{record['transaction_month']:02}/{record['transaction_year']:04}), '''
            f'''com uma contagem total de {int(record['transaction_value_count'])} transações '''
            f'''e valor total de R$ {record['transaction_value_sum']:.2f}, '''
            f'''o valor médio é de R$ {record['transaction_value_mean']:.2f}, '''
            f'''o valor da maior transaçao é de R$ {record['transaction_value_max']:.2f} e '''
            f'''o valor da menor transação é de R$ {record['transaction_value_min']:.2f}. '''
            f'''Dentre elas foram realizadas ''' + ', '.join(o for o in [
                f'''{int(record['card_variant_black_count'])} com cartão BLACK''' if int(record['card_variant_black_count']) > 0 else None,
                f'''{int(record['card_variant_gold_count'])} com cartão GOLD''' if int(record['card_variant_gold_count']) > 0 else None,
                f'''{int(record['card_variant_platinum_count'])} com cartão PLATINUM''' if int(record['card_variant_platinum_count']) > 0 else None,
                f'''{int(record['card_variant_standard_count'])} com cartão STANDARD''' if int(record['card_variant_standard_count']) > 0 else None,
                f'''{int(record['card_variant_international_count'])} com cartão INTERNACIONAL''' if int(record['card_variant_international_count']) > 0 else None
            ] if o) + '.'
    )

    result_dfs.append(CardTransactions.group_by_year_month_consumer(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Resumo mensal das transações com cartão de crédito do cliente {record['consumer_name']} (CPF: {record['consumer_document']}) '''
            f'''no mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_month']:02}/{record['transaction_year']:04}), '''
            f'''com uma contagem total de {int(record['transaction_value_count'])} transações '''
            f'''e valor total de R$ {record['transaction_value_sum']:.2f}, '''
            f'''o valor médio é de R$ {record['transaction_value_mean']:.2f}, '''
            f'''o valor da maior transaçao é de R$ {record['transaction_value_max']:.2f} e '''
            f'''o valor da menor transação é de R$ {record['transaction_value_min']:.2f}. '''
            f'''Dentre elas foram realizadas ''' + ', '.join(o for o in [
                f'''{int(record['card_variant_black_count'])} com cartão BLACK''' if int(record['card_variant_black_count']) > 0 else None,
                f'''{int(record['card_variant_gold_count'])} com cartão GOLD''' if int(record['card_variant_gold_count']) > 0 else None,
                f'''{int(record['card_variant_platinum_count'])} com cartão PLATINUM''' if int(record['card_variant_platinum_count']) > 0 else None,
                f'''{int(record['card_variant_standard_count'])} com cartão STANDARD''' if int(record['card_variant_standard_count']) > 0 else None,
                f'''{int(record['card_variant_international_count'])} com cartão INTERNACIONAL''' if int(record['card_variant_international_count']) > 0 else None
            ] if o) + '.'
    )

    result_dfs.append(CardTransactions.group_by_year_consumer(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Sumário anual de transações com cartão de crédito do cliente {record['consumer_name']} (CPF: {record['consumer_document']}) '''
            f'''no ano de {record['transaction_year']}, '''
            f'''com uma contagem total de {int(record['transaction_value_count'])} transações '''
            f'''e valor total de R$ {record['transaction_value_sum']:.2f}, '''
            f'''o valor médio é de R$ {record['transaction_value_mean']:.2f}, '''
            f'''o valor da maior transaçao é de R$ {record['transaction_value_max']:.2f} e '''
            f'''o valor da menor transação é de R$ {record['transaction_value_min']:.2f}. '''
            f'''Dentre elas foram realizadas ''' + ', '.join(o for o in [
                f'''{int(record['card_variant_black_count'])} com cartão BLACK''' if int(record['card_variant_black_count']) > 0 else None,
                f'''{int(record['card_variant_gold_count'])} com cartão GOLD''' if int(record['card_variant_gold_count']) > 0 else None,
                f'''{int(record['card_variant_platinum_count'])} com cartão PLATINUM''' if int(record['card_variant_platinum_count']) > 0 else None,
                f'''{int(record['card_variant_standard_count'])} com cartão STANDARD''' if int(record['card_variant_standard_count']) > 0 else None,
                f'''{int(record['card_variant_international_count'])} com cartão INTERNACIONAL''' if int(record['card_variant_international_count']) > 0 else None
            ] if o) + '.'
    )
    
    result_dfs.append(CardTransactions.group_by_year_month_day_portfolio(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Sumário diário de transações com cartão de crédito de todos os clientes da carteira "{record['portfolio_id']}" '''
            f'''no dia {record['transaction_day']} do mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_day']:02}/{record['transaction_month']:02}/{record['transaction_year']:04}), '''
            f'''com uma contagem total de {int(record['transaction_value_count'])} transações '''
            f'''e valor total de R$ {record['transaction_value_sum']:.2f}, '''
            f'''o valor médio é de R$ {record['transaction_value_mean']:.2f}, '''
            f'''o valor da maior transaçao é de R$ {record['transaction_value_max']:.2f} e '''
            f'''o valor da menor transação é de R$ {record['transaction_value_min']:.2f}. '''
            f'''Dentre elas foram realizadas ''' + ', '.join(o for o in [
                f'''{int(record['card_variant_black_count'])} com cartão BLACK''' if int(record['card_variant_black_count']) > 0 else None,
                f'''{int(record['card_variant_gold_count'])} com cartão GOLD''' if int(record['card_variant_gold_count']) > 0 else None,
                f'''{int(record['card_variant_platinum_count'])} com cartão PLATINUM''' if int(record['card_variant_platinum_count']) > 0 else None,
                f'''{int(record['card_variant_standard_count'])} com cartão STANDARD''' if int(record['card_variant_standard_count']) > 0 else None,
                f'''{int(record['card_variant_international_count'])} com cartão INTERNACIONAL''' if int(record['card_variant_international_count']) > 0 else None
            ] if o) + '.'
    )

    result_dfs.append(CardTransactions.group_by_year_month_portfolio(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Sumário mensal de transações com cartão de crédito de todos os clientes da carteira "{record['portfolio_id']}" '''
            f'''para o mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_month']:02}/{record['transaction_year']:04}), '''
            f'''com uma contagem total de {int(record['transaction_value_count'])} transações '''
            f'''e valor total de R$ {record['transaction_value_sum']:.2f}, '''
            f'''o valor médio é de R$ {record['transaction_value_mean']:.2f}, '''
            f'''o valor da maior transaçao é de R$ {record['transaction_value_max']:.2f} e '''
            f'''o valor da menor transação é de R$ {record['transaction_value_min']:.2f}. '''
            f'''Dentre elas foram realizadas ''' + ', '.join(o for o in [
                f'''{int(record['card_variant_black_count'])} com cartão BLACK''' if int(record['card_variant_black_count']) > 0 else None,
                f'''{int(record['card_variant_gold_count'])} com cartão GOLD''' if int(record['card_variant_gold_count']) > 0 else None,
                f'''{int(record['card_variant_platinum_count'])} com cartão PLATINUM''' if int(record['card_variant_platinum_count']) > 0 else None,
                f'''{int(record['card_variant_standard_count'])} com cartão STANDARD''' if int(record['card_variant_standard_count']) > 0 else None,
                f'''{int(record['card_variant_international_count'])} com cartão INTERNACIONAL''' if int(record['card_variant_international_count']) > 0 else None
            ] if o) + '.'
    )

    result_dfs.append(CardTransactions.group_by_year_portfolio(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Sumário anual de transações com cartão de crédito de todos os clientes da carteira "{record['portfolio_id']}" '''
            f'''para o ano de {record['transaction_year']}, '''
            f'''com uma contagem total de {int(record['transaction_value_count'])} transações '''
            f'''e valor total de R$ {record['transaction_value_sum']:.2f}, '''
            f'''o valor médio é de R$ {record['transaction_value_mean']:.2f}, '''
            f'''o valor da maior transaçao é de R$ {record['transaction_value_max']:.2f} e '''
            f'''o valor da menor transação é de R$ {record['transaction_value_min']:.2f}. '''
            f'''Dentre elas foram realizadas ''' + ', '.join(o for o in [
                f'''{int(record['card_variant_black_count'])} com cartão BLACK''' if int(record['card_variant_black_count']) > 0 else None,
                f'''{int(record['card_variant_gold_count'])} com cartão GOLD''' if int(record['card_variant_gold_count']) > 0 else None,
                f'''{int(record['card_variant_platinum_count'])} com cartão PLATINUM''' if int(record['card_variant_platinum_count']) > 0 else None,
                f'''{int(record['card_variant_standard_count'])} com cartão STANDARD''' if int(record['card_variant_standard_count']) > 0 else None,
                f'''{int(record['card_variant_international_count'])} com cartão INTERNACIONAL''' if int(record['card_variant_international_count']) > 0 else None
            ] if o) + '.'
    )

    df = concat(result_dfs)
    vectorstore_name = 'prj-ai-rag-llm-table-10-granularity'
    embedding_model_name = 'text-embedding-3-small'
    if insert:
        write_documents_txt(vectorstore_name, documents)
        insert_documents(vectorstore_name, documents, embedding_model_name)
    return df, documents


# Use large text embedding model
def test_11(df: DataFrame, aggregations: Dict[str, Tuple[str, Any]], insert: bool = False) -> Tuple[DataFrame, List[Document]]:
    result_dfs = []
    documents = []

    df = df[df['product'] == 'crédito']

    result_dfs.append(CardTransactions.group_by_transaction(df))
    documents = generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''O cliente {record['consumer_name']} (CPF: {record['consumer_document']}) '''
            f'''efetuou a transação de ID "{record['transaction_id']}" com cartão de crédito {record['card_variant']} de R$ {record['transaction_value']:.2f} '''
            f'''no dia {record['transaction_day']} do mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_day']:02}/{record['transaction_month']:02}/{record['transaction_year']:04}) '''
            f'''para o estabelecimento "{record['seller_description']}"'''
    )
    
    result_dfs.append(CardTransactions.group_by_year_month_day_consumer(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Resumo diário das transações com cartão de crédito do cliente {record['consumer_name']} (CPF: {record['consumer_document']}) '''
            f'''no dia {record['transaction_day']} do mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_day']:02}/{record['transaction_month']:02}/{record['transaction_year']:04}), '''
            f'''com uma contagem total de {int(record['transaction_value_count'])} transações '''
            f'''e valor total de R$ {record['transaction_value_sum']:.2f}, '''
            f'''o valor médio é de R$ {record['transaction_value_mean']:.2f}, '''
            f'''o valor da maior transaçao é de R$ {record['transaction_value_max']:.2f} e '''
            f'''o valor da menor transação é de R$ {record['transaction_value_min']:.2f}. '''
            f'''Dentre elas foram realizadas ''' + ', '.join(o for o in [
                f'''{int(record['card_variant_black_count'])} com cartão BLACK''' if int(record['card_variant_black_count']) > 0 else None,
                f'''{int(record['card_variant_gold_count'])} com cartão GOLD''' if int(record['card_variant_gold_count']) > 0 else None,
                f'''{int(record['card_variant_platinum_count'])} com cartão PLATINUM''' if int(record['card_variant_platinum_count']) > 0 else None,
                f'''{int(record['card_variant_standard_count'])} com cartão STANDARD''' if int(record['card_variant_standard_count']) > 0 else None,
                f'''{int(record['card_variant_international_count'])} com cartão INTERNACIONAL''' if int(record['card_variant_international_count']) > 0 else None
            ] if o) + '.'
    )

    result_dfs.append(CardTransactions.group_by_year_month_consumer(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Resumo mensal das transações com cartão de crédito do cliente {record['consumer_name']} (CPF: {record['consumer_document']}) '''
            f'''no mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_month']:02}/{record['transaction_year']:04}), '''
            f'''com uma contagem total de {int(record['transaction_value_count'])} transações '''
            f'''e valor total de R$ {record['transaction_value_sum']:.2f}, '''
            f'''o valor médio é de R$ {record['transaction_value_mean']:.2f}, '''
            f'''o valor da maior transaçao é de R$ {record['transaction_value_max']:.2f} e '''
            f'''o valor da menor transação é de R$ {record['transaction_value_min']:.2f}. '''
            f'''Dentre elas foram realizadas ''' + ', '.join(o for o in [
                f'''{int(record['card_variant_black_count'])} com cartão BLACK''' if int(record['card_variant_black_count']) > 0 else None,
                f'''{int(record['card_variant_gold_count'])} com cartão GOLD''' if int(record['card_variant_gold_count']) > 0 else None,
                f'''{int(record['card_variant_platinum_count'])} com cartão PLATINUM''' if int(record['card_variant_platinum_count']) > 0 else None,
                f'''{int(record['card_variant_standard_count'])} com cartão STANDARD''' if int(record['card_variant_standard_count']) > 0 else None,
                f'''{int(record['card_variant_international_count'])} com cartão INTERNACIONAL''' if int(record['card_variant_international_count']) > 0 else None
            ] if o) + '.'
    )

    result_dfs.append(CardTransactions.group_by_year_consumer(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Sumário anual de transações com cartão de crédito do cliente {record['consumer_name']} (CPF: {record['consumer_document']}) '''
            f'''no ano de {record['transaction_year']}, '''
            f'''com uma contagem total de {int(record['transaction_value_count'])} transações '''
            f'''e valor total de R$ {record['transaction_value_sum']:.2f}, '''
            f'''o valor médio é de R$ {record['transaction_value_mean']:.2f}, '''
            f'''o valor da maior transaçao é de R$ {record['transaction_value_max']:.2f} e '''
            f'''o valor da menor transação é de R$ {record['transaction_value_min']:.2f}. '''
            f'''Dentre elas foram realizadas ''' + ', '.join(o for o in [
                f'''{int(record['card_variant_black_count'])} com cartão BLACK''' if int(record['card_variant_black_count']) > 0 else None,
                f'''{int(record['card_variant_gold_count'])} com cartão GOLD''' if int(record['card_variant_gold_count']) > 0 else None,
                f'''{int(record['card_variant_platinum_count'])} com cartão PLATINUM''' if int(record['card_variant_platinum_count']) > 0 else None,
                f'''{int(record['card_variant_standard_count'])} com cartão STANDARD''' if int(record['card_variant_standard_count']) > 0 else None,
                f'''{int(record['card_variant_international_count'])} com cartão INTERNACIONAL''' if int(record['card_variant_international_count']) > 0 else None
            ] if o) + '.'
    )
    
    result_dfs.append(CardTransactions.group_by_year_month_day_portfolio(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Sumário diário de transações com cartão de crédito de todos os clientes da carteira "{record['portfolio_id']}" '''
            f'''no dia {record['transaction_day']} do mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_day']:02}/{record['transaction_month']:02}/{record['transaction_year']:04}), '''
            f'''com uma contagem total de {int(record['transaction_value_count'])} transações '''
            f'''e valor total de R$ {record['transaction_value_sum']:.2f}, '''
            f'''o valor médio é de R$ {record['transaction_value_mean']:.2f}, '''
            f'''o valor da maior transaçao é de R$ {record['transaction_value_max']:.2f} e '''
            f'''o valor da menor transação é de R$ {record['transaction_value_min']:.2f}. '''
            f'''Dentre elas foram realizadas ''' + ', '.join(o for o in [
                f'''{int(record['card_variant_black_count'])} com cartão BLACK''' if int(record['card_variant_black_count']) > 0 else None,
                f'''{int(record['card_variant_gold_count'])} com cartão GOLD''' if int(record['card_variant_gold_count']) > 0 else None,
                f'''{int(record['card_variant_platinum_count'])} com cartão PLATINUM''' if int(record['card_variant_platinum_count']) > 0 else None,
                f'''{int(record['card_variant_standard_count'])} com cartão STANDARD''' if int(record['card_variant_standard_count']) > 0 else None,
                f'''{int(record['card_variant_international_count'])} com cartão INTERNACIONAL''' if int(record['card_variant_international_count']) > 0 else None
            ] if o) + '.'
    )

    result_dfs.append(CardTransactions.group_by_year_month_portfolio(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Sumário mensal de transações com cartão de crédito de todos os clientes da carteira "{record['portfolio_id']}" '''
            f'''para o mês de {get_month_name(record['transaction_month'])} do ano de {record['transaction_year']} '''
            f'''({record['transaction_month']:02}/{record['transaction_year']:04}), '''
            f'''com uma contagem total de {int(record['transaction_value_count'])} transações '''
            f'''e valor total de R$ {record['transaction_value_sum']:.2f}, '''
            f'''o valor médio é de R$ {record['transaction_value_mean']:.2f}, '''
            f'''o valor da maior transaçao é de R$ {record['transaction_value_max']:.2f} e '''
            f'''o valor da menor transação é de R$ {record['transaction_value_min']:.2f}. '''
            f'''Dentre elas foram realizadas ''' + ', '.join(o for o in [
                f'''{int(record['card_variant_black_count'])} com cartão BLACK''' if int(record['card_variant_black_count']) > 0 else None,
                f'''{int(record['card_variant_gold_count'])} com cartão GOLD''' if int(record['card_variant_gold_count']) > 0 else None,
                f'''{int(record['card_variant_platinum_count'])} com cartão PLATINUM''' if int(record['card_variant_platinum_count']) > 0 else None,
                f'''{int(record['card_variant_standard_count'])} com cartão STANDARD''' if int(record['card_variant_standard_count']) > 0 else None,
                f'''{int(record['card_variant_international_count'])} com cartão INTERNACIONAL''' if int(record['card_variant_international_count']) > 0 else None
            ] if o) + '.'
    )

    result_dfs.append(CardTransactions.group_by_year_portfolio(df, aggregations))
    documents += generate_documents(
        df=result_dfs[-1],
        parse_content_header=lambda record:
            f'''Sumário anual de transações com cartão de crédito de todos os clientes da carteira "{record['portfolio_id']}" '''
            f'''para o ano de {record['transaction_year']}, '''
            f'''com uma contagem total de {int(record['transaction_value_count'])} transações '''
            f'''e valor total de R$ {record['transaction_value_sum']:.2f}, '''
            f'''o valor médio é de R$ {record['transaction_value_mean']:.2f}, '''
            f'''o valor da maior transaçao é de R$ {record['transaction_value_max']:.2f} e '''
            f'''o valor da menor transação é de R$ {record['transaction_value_min']:.2f}. '''
            f'''Dentre elas foram realizadas ''' + ', '.join(o for o in [
                f'''{int(record['card_variant_black_count'])} com cartão BLACK''' if int(record['card_variant_black_count']) > 0 else None,
                f'''{int(record['card_variant_gold_count'])} com cartão GOLD''' if int(record['card_variant_gold_count']) > 0 else None,
                f'''{int(record['card_variant_platinum_count'])} com cartão PLATINUM''' if int(record['card_variant_platinum_count']) > 0 else None,
                f'''{int(record['card_variant_standard_count'])} com cartão STANDARD''' if int(record['card_variant_standard_count']) > 0 else None,
                f'''{int(record['card_variant_international_count'])} com cartão INTERNACIONAL''' if int(record['card_variant_international_count']) > 0 else None
            ] if o) + '.'
    )

    df = concat(result_dfs)
    vectorstore_name = 'prj-ai-rag-llm-table-11-large-embedding'
    embedding_model_name = 'text-embedding-3-large'
    if insert:
        write_documents_txt(vectorstore_name, documents)
        insert_documents(vectorstore_name, documents, embedding_model_name)
    return df, documents


if __name__ == '__main__':
    df = CardTransactions.read()
    df = CardTransactions.refine(df)
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
        'card_variant_international_count': ('card_variant', lambda x: x.value_counts().loc['INTERNACIONAL']),
    }

    tests = []
    # # 'prj-ai-rag-llm-table-1-standard'
    # tests.append(test_1(df, aggregations, insert=True))  # Standard processing
    # # 'prj-ai-rag-llm-table-2-discursive'
    # tests.append(test_2(df, aggregations, insert=True))  # Discursive text content
    # # 'prj-ai-rag-llm-table-3-standard-creditcard'
    # tests.append(test_3(df, aggregations, insert=True))  # Process credit cards data only
    # # 'prj-ai-rag-llm-table-4-discursive-creditcard'
    # tests.append(test_4(df, aggregations, insert=True))  # Discursive text content and only credit cards data
    # # 'prj-ai-rag-llm-table-5-join-wo-transactions'
    # tests.append(test_5(df, aggregations, insert=True))  # Join transactions with diary summaries
    # # 'prj-ai-rag-llm-table-6-chunks'
    # tests.append(test_6(df, aggregations, insert=True))  # Chunks of 1000 tokens
    # # 'prj-ai-rag-llm-table-7-cohere'
    # tests.append(test_7(df, aggregations, insert=True))  # Cohere embeddings
    # # 'prj-ai-rag-llm-table-8-storytelling'
    # tests.append(test_8(df, aggregations, insert=True))  # Day becomes "no dia", Month becomes "resumo", Year becomes "sumário"
    # # 'prj-ai-rag-llm-table-9-questions'
    # tests.append(test_9(df, aggregations, insert=True))  # Questions included in the document
    # # 'prj-ai-rag-llm-table-10-granularity'
    # tests.append(test_10(df, aggregations, insert=True))  # Store granularity in metadata
    # # 'prj-ai-rag-llm-table-11-large-embedding'
    tests.append(test_11(df, aggregations, insert=True))  # Use large text embedding model
