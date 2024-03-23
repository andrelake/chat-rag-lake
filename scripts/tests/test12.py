'''
Test 12 - Embedding documents in english
'''


import os

from env import PINECONE_API_KEY, OPENAI_API_KEY
from utils import log, get_month_name
from data_utils.handlers import DocumentsHandler
from data_utils.tables import CardTransactions
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
    model_name = 'text-embedding-3-large'
    type = 'api'
    client = get_client(model_name=model_name, type=type, api_key=OPENAI_API_KEY)
    dimension_count = get_dimension_count(model_name=model_name, type=type)


class Database:
    client = get_database_client(PINECONE_API_KEY)
    vectorstore_name = 'prj-ai-rag-llm-table-12-english'


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
documents = DocumentsHandler.from_dataframe(
    df=result_dfs[-1],
    parse_content_header=lambda record:
        f'''Customer {record['consumer_name']} (CPF: {record['consumer_document']}) '''
        f'''made a credit card ({record['card_variant']}) transaction of R$ {record['transaction_value']:.2f} (ID "{record['transaction_id']}") '''
        f'''on {get_month_name(record['transaction_month'])} {record['transaction_day']}, {record['transaction_year']} '''
        f'''({record['transaction_day']:02}/{record['transaction_month']:02}/{record['transaction_year']:04}) '''
        f'''at the establishment "{record['seller_description']}"''',
)

result_dfs.append(CardTransactions.group_by_year_month_day_consumer(df, aggregations))
documents += DocumentsHandler.from_dataframe(
    df=result_dfs[-1],
    parse_content_header=lambda record:
        f'''Daily summary of credit card transactions for customer {record['consumer_name']} (CPF: {record['consumer_document']}) '''
        f'''on {get_month_name(record['transaction_month'])} {record['transaction_day']}, {record['transaction_year']}, '''
        f'''({record['transaction_day']:02}/{record['transaction_month']:02}/{record['transaction_year']:04}), '''
        f'''with a total count of {int(record['transaction_value_count'])} transactions and a total value of R$ {record['transaction_value_sum']:.2f}, '''
        f'''the average value is R$ {record['transaction_value_mean']:.2f}, the value of the largest transaction is R$ {record['transaction_value_max']:.2f} and '''
        f'''the value of the smallest transaction is R$ {record['transaction_value_min']:.2f}. '''
        f'''Of these, ''' + ', '.join(o for o in [
            f'''{int(record['card_variant_black_count'])} were made with a BLACK card''' if int(record['card_variant_black_count']) > 0 else None,
            f'''{int(record['card_variant_gold_count'])} were made with a GOLD card''' if int(record['card_variant_gold_count']) > 0 else None,
            f'''{int(record['card_variant_platinum_count'])} were made with a PLATINUM card''' if int(record['card_variant_platinum_count']) > 0 else None,
            f'''{int(record['card_variant_standard_count'])} were made with a STANDARD card''' if int(record['card_variant_standard_count']) > 0 else None,
            f'''{int(record['card_variant_international_count'])} were made with an INTERNATIONAL card''' if int(record['card_variant_international_count']) > 0 else None
        ] if o) + '.'
)

result_dfs.append(CardTransactions.group_by_year_month_consumer(df, aggregations))
documents += DocumentsHandler.from_dataframe(
    df=result_dfs[-1],
    parse_content_header=lambda record:
        f'''Monthly summary of credit card transactions for customer {record['consumer_name']} (CPF: {record['consumer_document']}) '''
        f'''for the entire month of {get_month_name(record['transaction_month'])}, {record['transaction_year']} '''
        f'''({record['transaction_month']:02}/{record['transaction_year']:04}), '''
        f'''with a total count of {int(record['transaction_value_count'])} transactions and a total value of R$ {record['transaction_value_sum']:.2f}, '''
        f'''the average value is R$ {record['transaction_value_mean']:.2f}, the value of the largest transaction is R$ {record['transaction_value_max']:.2f} and '''
        f'''the value of the smallest transaction is R$ {record['transaction_value_min']:.2f}. '''
        f'''Of these, ''' + ', '.join(o for o in [
            f'''{int(record['card_variant_black_count'])} were made with a BLACK card''' if int(record['card_variant_black_count']) > 0 else None,
            f'''{int(record['card_variant_gold_count'])} were made with a GOLD card''' if int(record['card_variant_gold_count']) > 0 else None,
            f'''{int(record['card_variant_platinum_count'])} were made with a PLATINUM card''' if int(record['card_variant_platinum_count']) > 0 else None,
            f'''{int(record['card_variant_standard_count'])} were made with a STANDARD card''' if int(record['card_variant_standard_count']) > 0 else None,
            f'''{int(record['card_variant_international_count'])} were made with an INTERNATIONAL card''' if int(record['card_variant_international_count']) > 0 else None
        ] if o) + '.'
)

result_dfs.append(CardTransactions.group_by_year_consumer(df, aggregations))
documents += DocumentsHandler.from_dataframe(
    df=result_dfs[-1],
    parse_content_header=lambda record:
        f'''Annual summary of credit card transactions for customer {record['consumer_name']} (CPF: {record['consumer_document']}) '''
        f'''for the entire year of {record['transaction_year']}, '''
        f'''with a total count of {int(record['transaction_value_count'])} transactions and a total value of R$ {record['transaction_value_sum']:.2f}, '''
        f'''the average value is R$ {record['transaction_value_mean']:.2f}, the value of the largest transaction is R$ {record['transaction_value_max']:.2f} and '''
        f'''the value of the smallest transaction is R$ {record['transaction_value_min']:.2f}. '''
        f'''Of these, ''' + ', '.join(o for o in [
            f'''{int(record['card_variant_black_count'])} were made with a BLACK card''' if int(record['card_variant_black_count']) > 0 else None,
            f'''{int(record['card_variant_gold_count'])} were made with a GOLD card''' if int(record['card_variant_gold_count']) > 0 else None,
            f'''{int(record['card_variant_platinum_count'])} were made with a PLATINUM card''' if int(record['card_variant_platinum_count']) > 0 else None,
            f'''{int(record['card_variant_standard_count'])} were made with a STANDARD card''' if int(record['card_variant_standard_count']) > 0 else None,
            f'''{int(record['card_variant_international_count'])} were made with an INTERNATIONAL card''' if int(record['card_variant_international_count']) > 0 else None
        ] if o) + '.'
)

result_dfs.append(CardTransactions.group_by_year_month_day_portfolio(df, aggregations))
documents += DocumentsHandler.from_dataframe(
    df=result_dfs[-1],
    parse_content_header=lambda record:
        f'''Daily summary of credit card transactions for all customers of the portfolio "{record['portfolio_id']}" '''
        f'''on {get_month_name(record['transaction_month'])} {record['transaction_day']}, {record['transaction_year']}, '''
        f'''({record['transaction_day']:02}/{record['transaction_month']:02}/{record['transaction_year']:04}), '''
        f'''with a total count of {int(record['transaction_value_count'])} transactions and a total value of R$ {record['transaction_value_sum']:.2f}, '''
        f'''the average value is R$ {record['transaction_value_mean']:.2f}, the value of the largest transaction is R$ {record['transaction_value_max']:.2f} and '''
        f'''the value of the smallest transaction is R$ {record['transaction_value_min']:.2f}. '''
        f'''Of these, ''' + ', '.join(o for o in [
            f'''{int(record['card_variant_black_count'])} were made with a BLACK card''' if int(record['card_variant_black_count']) > 0 else None,
            f'''{int(record['card_variant_gold_count'])} were made with a GOLD card''' if int(record['card_variant_gold_count']) > 0 else None,
            f'''{int(record['card_variant_platinum_count'])} were made with a PLATINUM card''' if int(record['card_variant_platinum_count']) > 0 else None,
            f'''{int(record['card_variant_standard_count'])} were made with a STANDARD card''' if int(record['card_variant_standard_count']) > 0 else None,
            f'''{int(record['card_variant_international_count'])} were made with an INTERNATIONAL card''' if int(record['card_variant_international_count']) > 0 else None
        ] if o) + '.'
        
)

result_dfs.append(CardTransactions.group_by_year_month_portfolio(df, aggregations))
documents += DocumentsHandler.from_dataframe(
    df=result_dfs[-1],
    parse_content_header=lambda record:
        f'''Monthly summary of credit card transactions for all customers of the portfolio "{record['portfolio_id']}" '''
        f'''for the entire month of {get_month_name(record['transaction_month'])}, {record['transaction_year']} '''
        f'''({record['transaction_month']:02}/{record['transaction_year']:04}), '''
        f'''with a total count of {int(record['transaction_value_count'])} transactions and a total value of R$ {record['transaction_value_sum']:.2f}, '''
        f'''the average value is R$ {record['transaction_value_mean']:.2f}, the value of the largest transaction is R$ {record['transaction_value_max']:.2f} and '''
        f'''the value of the smallest transaction is R$ {record['transaction_value_min']:.2f}. '''
        f'''Of these, ''' + ', '.join(o for o in [
            f'''{int(record['card_variant_black_count'])} were made with a BLACK card''' if int(record['card_variant_black_count']) > 0 else None,
            f'''{int(record['card_variant_gold_count'])} were made with a GOLD card''' if int(record['card_variant_gold_count']) > 0 else None,
            f'''{int(record['card_variant_platinum_count'])} were made with a PLATINUM card''' if int(record['card_variant_platinum_count']) > 0 else None,
            f'''{int(record['card_variant_standard_count'])} were made with a STANDARD card''' if int(record['card_variant_standard_count']) > 0 else None,
            f'''{int(record['card_variant_international_count'])} were made with an INTERNATIONAL card''' if int(record['card_variant_international_count']) > 0 else None
        ] if o) + '.'
)

result_dfs.append(CardTransactions.group_by_year_portfolio(df, aggregations))
documents += DocumentsHandler.from_dataframe(
    df=result_dfs[-1],
    parse_content_header=lambda record:
        f'''Annual summary of credit card transactions for all customers of the portfolio "{record['portfolio_id']}" '''
        f'''for the entire year of {record['transaction_year']}, '''
        f'''with a total count of {int(record['transaction_value_count'])} transactions and a total value of R$ {record['transaction_value_sum']:.2f}, '''
        f'''the average value is R$ {record['transaction_value_mean']:.2f}, the value of the largest transaction is R$ {record['transaction_value_max']:.2f} and '''
        f'''the value of the smallest transaction is R$ {record['transaction_value_min']:.2f}. '''
        f'''Of these, ''' + ', '.join(o for o in [
            f'''{int(record['card_variant_black_count'])} were made with a BLACK card''' if int(record['card_variant_black_count']) > 0 else None,
            f'''{int(record['card_variant_gold_count'])} were made with a GOLD card''' if int(record['card_variant_gold_count']) > 0 else None,
            f'''{int(record['card_variant_platinum_count'])} were made with a PLATINUM card''' if int(record['card_variant_platinum_count']) > 0 else None,
            f'''{int(record['card_variant_standard_count'])} were made with a STANDARD card''' if int(record['card_variant_standard_count']) > 0 else None,
            f'''{int(record['card_variant_international_count'])} were made with an INTERNATIONAL card''' if int(record['card_variant_international_count']) > 0 else None
        ] if o) + '.'
)

df = concat(result_dfs)

if __name__ == '__main__':
    DocumentsHandler.write_txt(os.path.join('data', 'refined', 'pinecone', Database.vectorstore_name, 'data.txt'), documents)

    # Recreate the vectorstore
    delete_vectorstore(name=Database.vectorstore_name, database_client=Database.client)
    Database.vectorstore = get_vectorstore(
        name=Database.vectorstore_name,
        embedding_function=Embedding.client,
        database_client=Database.client,
        create=True,
        dimension_count=Embedding.dimension_count
    )

    # Add documents to the vectorstore
    get_cost(documents=documents, model_name=Embedding.model_name, type=Embedding.type)
    add_documents(
        vectorstore=Database.vectorstore,
        documents=documents,
        embedding_function=Embedding.client,
        vectorstore_name=Database.vectorstore_name,
    )
    log(Database.vectorstore._index.describe_index_stats())
