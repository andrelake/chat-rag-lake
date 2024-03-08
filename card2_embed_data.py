import os
from env import PINECONE_API_KEY, OPENAI_API_KEY
from datetime import date
from typing import Optional, Any, Callable, List, Dict, Tuple
import json

from utils import log, get_month_name
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

from pandas import DataFrame, concat
from langchain_core.documents import Document


# Configure Logger
log.verbose = True
log.end = '\n\n'

# Get database client
database_client = get_database_client(api_key=PINECONE_API_KEY)
database_type = 'pinecone'

# Get embeddings client
embedding_model_name = 'text-embedding-3-small'
embedding_function = get_embeddings_client(model_name=embedding_model_name, type='api', api_key=OPENAI_API_KEY)


def insert_documents(vectorstore_name: str, documents: List[Document]) -> None:
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

def write_documents_txt(vectorstore_name: str, documents: List[Document]) -> None:
    text_path = os.path.join('data', 'refined', database_type, vectorstore_name)
    os.makedirs(text_path, exist_ok=True)
    with open(os.path.join(text_path, 'data.txt'), 'w') as fo:
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
    if insert:
        write_documents_txt(vectorstore_name, documents)
        insert_documents(vectorstore_name, documents)
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
    if insert:
        write_documents_txt(vectorstore_name, documents)
        insert_documents(vectorstore_name, documents)
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
    if insert:
        write_documents_txt(vectorstore_name, documents)
        insert_documents(vectorstore_name, documents)
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
            f'''{record['card_variant']} para o estabelecimento "{record['seller_description']}"'''
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
    if insert:
        write_documents_txt(vectorstore_name, documents)
        insert_documents(vectorstore_name, documents)
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
        'card_variant_international_count': ('card_variant', lambda x: x.value_counts().loc['INTERNACIONAL'])
    }

    tests = []
    tests.append(test_1(df, aggregations, insert=True))
    tests.append(test_2(df, aggregations, insert=True))
    tests.append(test_3(df, aggregations, insert=True))
    tests.append(test_4(df, aggregations, insert=True))
