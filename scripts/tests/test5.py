'''
Test 5 - Join transactions with daily summaries
'''


from env import PINECONE_API_KEY, OPENAI_API_KEY
from utils import log, get_month_name
from data_handler import DocumentsHandler
from data_tables import CardTransactions
from connections.embeddings import get_client, get_cost, get_dimension_count
from connections.pinecone import (
    get_database_client,
    get_vectorstore,
    delete_vectorstore,
    add_documents,
)

from pandas import concat


# Configure Logger
log.verbose = True
log.end = '\n\n'


class Embedding:
    model_name = 'text-embedding-3-small'
    type = 'api'
    client = get_client(model_name=model_name, type=type, api_key=OPENAI_API_KEY)
    dimension_count = get_dimension_count(model_name=model_name, type=type)


class Database:
    client = get_database_client(PINECONE_API_KEY)
    vectorstore_name = 'prj-ai-rag-llm-table-5-join-wo-transactions'
    vectorstore = get_vectorstore(
        name=vectorstore_name,
        embedding_function=Embedding.client,
        database_client=client,
        create=True,
        dimension_count=Embedding.dimension_count
    )


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


result_dfs = []
documents = []

df = df[df['product'] == 'crédito']

result_dfs.append(CardTransactions.group_by_transaction(df))
documents += DocumentsHandler.from_dataframe(
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
documents += DocumentsHandler.from_dataframe(
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
documents += DocumentsHandler.from_dataframe(
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
documents += DocumentsHandler.from_dataframe(
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
documents += DocumentsHandler.from_dataframe(
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
documents += DocumentsHandler.from_dataframe(
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

if __name__ == '__main__':
    DocumentsHandler.write_txt(Database.vectorstore_name, documents)

    # Add documents
    delete_vectorstore(name=Database.vectorstore_name, database_client=Database.client)
    get_cost(documents=documents, model_name=Embedding.model_name)
    add_documents(
        vectorstore=Database.vectorstore,
        documents=documents,
        embedding_function=Embedding.client,
        vectorstore_name=Database.vectorstore_name,
    )
    log(Database.vectorstore._index.describe_index_stats())
