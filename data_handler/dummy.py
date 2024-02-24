import os
from typing import List
import pytz
import random
from datetime import date

import pandas as pd
from faker import Faker
from fastavro import writer, parse_schema


class Logger:
    def __init__(self, end: str = '\n', verbose: bool = False, **kwargs):
        self.end = end
        self.verbose = verbose
        self.kwargs = kwargs

    def __call__(self, message: str, **kwargs):
        kwargs = {'end': self.end, **self.kwargs, **kwargs}
        if self.verbose:
            print(message, **kwargs)


log = Logger()


def generate_dummy_data(
    group_by: List,
    n_officers: int,
    n_consumers_officer: int,
    n_transactions_consumer_day: int,
    start_date: date,
    end_date: date,
    chaos_consumers_officer: float = 0,
    chaos_transactions_client_day: float = 0,
    log: callable = log,
    save_path: str = None
):

    # PySpark data schema
    pyspark_schema = '''
        transaction_id bigint,
        transaction_at timestamp,
        transaction_year smallint,
        transaction_month byte,
        transaction_day byte,
        consumer_id_hash bigint,
        consumer_id bigint,
        consumer_document string,
        consumer_name string,
        portfolio_id bigint,
        officer_id bigint,
        officer_name string,
        product string,
        card_variant string,
        transaction_value double,
        seller_description string
    '''

    pandas_schema = {
        'transaction_id': pd.UInt32Dtype(),
        'transaction_at': pd.StringDtype(),  # pd.DatetimeTZDtype(tz='UTC'),
        'transaction_year': pd.Int16Dtype(),
        'transaction_month': pd.UInt8Dtype(),
        'transaction_day': pd.UInt8Dtype(),
        'consumer_id_hash': pd.UInt32Dtype(),
        'consumer_id': pd.UInt32Dtype(),
        'consumer_document': pd.StringDtype(),
        'consumer_name': pd.StringDtype(),
        'portfolio_id': pd.UInt32Dtype(),
        'officer_id': pd.UInt32Dtype(),
        'officer_name': pd.StringDtype(),
        'product': pd.CategoricalDtype(categories=['', 'credit', 'debit'], ordered=True),
        'card_variant': pd.CategoricalDtype(categories=['', 'black', 'gold', 'platinum', 'standard'], ordered=True),
        'transaction_value': pd.Float64Dtype(),
        'seller_description': pd.StringDtype()
    }

    avro_schema = parse_schema({
        'type': 'record',
        'name': 'Transaction',
        'fields': [
            {'name': 'transaction_id', 'type': 'long'},
            {'name': 'transaction_at', 'type': 'string'},
            {'name': 'transaction_year', 'type': 'int'},
            {'name': 'transaction_month', 'type': 'int'},
            {'name': 'transaction_day', 'type': 'int'},
            {'name': 'consumer_id_hash', 'type': 'long'},
            {'name': 'consumer_id', 'type': 'long'},
            {'name': 'consumer_document', 'type': 'string'},
            {'name': 'consumer_name', 'type': 'string'},
            {'name': 'portfolio_id', 'type': 'long'},
            {'name': 'officer_id', 'type': 'long'},
            {'name': 'officer_name', 'type': 'string'},
            {'name': 'product', 'type': ['null', 'string'], 'default': None},
            {'name': 'card_variant', 'type': ['null', 'string'], 'default': None},
            {'name': 'transaction_value', 'type': 'double'},
            {'name': 'seller_description', 'type': 'string'}
        ]
    })

    # Generate data into a pandas dataframe
    fake = Faker(locale='pt_BR')
    log(f'Generating dummy data for `{n_officers}` officers and `{n_consumers_officer}` consumers per officer...')
    officers = [(i, fake.name(), i) for i in range(n_officers)]

    # Generate data
    log(f'Generating transactions...')
    dfs = []
    data_chunk = []
    i_transaction_id = 0
    for portfolio_id, officer_name, officer_id in officers:
        count_transactions_portfolio = 0
        for consumer_id in range(int(n_consumers_officer * (1 + random.uniform(-chaos_consumers_officer, chaos_consumers_officer)))):
            data_chunk = []
            consumer_document = fake.cpf()
            consumer_name = fake.name()
            for transaction_date in pd.date_range(start=start_date, end=end_date, freq='D'):
                for _ in range(int(n_transactions_consumer_day * (1 + random.uniform(-chaos_transactions_client_day, chaos_transactions_client_day)))):
                    transaction_at = transaction_date + pd.Timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59), seconds=random.randint(0, 59))
                    transaction_at = transaction_at.replace(tzinfo=pytz.utc)
                    data_chunk.append({
                        'transaction_id': i_transaction_id,
                        'transaction_at': transaction_at.strftime('%Y-%m-%dT%H:%M:%S.%f%z'),
                        'transaction_year': transaction_date.year,
                        'transaction_month': transaction_date.month,
                        'transaction_day': transaction_date.day,
                        'consumer_id_hash': consumer_id,
                        'consumer_id': consumer_id,
                        'consumer_document': consumer_document,
                        'consumer_name': consumer_name,
                        'portfolio_id': portfolio_id,
                        'officer_id': officer_id,
                        'officer_name': officer_name,
                        'product': random.choice(['', 'credit', 'debit']),
                        'card_variant': random.choice(['', 'black', 'gold', 'platinum', 'standard']),
                        'transaction_value': random.uniform(1, 5000),
                        'seller_description': fake.company()
                    })
            if data_chunk:
                count_transactions_portfolio += len(data_chunk)
                df = pd.DataFrame(data=data_chunk, columns=pandas_schema.keys()).astype(pandas_schema)
                dfs.append(df)
        log(f'\tGenerated `{count_transactions_portfolio}` transactions'
            f'for portfolio `{portfolio_id}` (officer `{officer_id}`: `{officer_name}`).',
            end='\n')
    
    df = pd.concat(dfs, ignore_index=True)

    log(f'\nGenerated `{len(df)}` transactions in total.')

    # Create sorted index
    log(f'Sorting & reindexing data by `{group_by}`...')
    df.set_index(group_by, inplace=True, drop=False)
    df.sort_index(ascending=True, inplace=True)

    # Dataframe info, size and memory usage
    log(f'Dataframe info:')
    log(df.info(), end='\n')
    log(f'Dataframe size: `{df.shape}`', end='\n')
    log(f'Dataframe memory usage: `{df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB`')

    if save_path:
        # Save data to disk as Avro partitioned by year
        for group_name, group_df in df.groupby(level=0):
            path = os.path.join(save_path, f'ptt_transaction_year={group_name}', 'data.avro')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                writer(f, avro_schema, group_df.to_dict(orient='records'))

            from pprint import pprint
            group_df[:1].to_dict(orient='records')

            log(f'Saved `{path}` with `{len(group_df)}` records.', end='\n')
    
    return df


if __name__ == '__main__':

    log.verbose = True
    log.end = '\n\n'

    df = generate_dummy_data(
        group_by=[
            'transaction_year',
            'portfolio_id',
            'consumer_id',
        ],
        n_officers=1,
        n_consumers_officer=10,
        n_transactions_consumer_day=6,
        start_date=date(2020, 1, 1),
        end_date=date(2024, 2, 29),
        chaos_consumers_officer=0.5,
        chaos_transactions_client_day=0.5,
        log=log,
        save_path=os.path.join('data', 'card_transactions')
    )

    # Prompt some complexbusiness questions
    officer_name = df.iloc[random.randint(0, len(df))].officer_name
    log(f'Query: Qual o valor total de transações em janeiro de 2023 feitas pelos clientes da carteira do gerente {officer_name}?', end='\n')
    result = df[df.transaction_year == 2023 & df.transaction_month == 1 & df.officer_name == officer_name].transaction_value.sum()
    log(f'Correct answer: `{result}`.')

    log(f'Query: Quantos clientes possuem cartão de crédito e débito?', end='\n')

