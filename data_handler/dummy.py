import os

import pandas as pd
from faker import Faker
import random
from fastavro import writer, parse_schema, reader
from datetime import date


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
    n_officers: int,
    n_consumers_officer: int,
    n_transactions_consumer_day: int,
    start_date: date,
    end_date: date,
    chaos_consumers_officer: float,
    chaos_transactions_client_day: float,
    log: callable
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
        'transaction_at': pd.DatetimeTZDtype(),
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

    # Generate data into a pandas dataframe
    fake = Faker()
    log(f'Generating dummy data for `{n_officers}` officers and `{n_consumers_officer}` consumers per officer...')
    officers = [(i, fake.name(), i) for i in range(n_officers)]

    # Generate data
    log(f'Generating transactions...')
    df = pd.DataFrame(columns=pandas_schema.keys()).astype(pandas_schema)
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
                    data_chunk.append({
                        'transaction_id': i_transaction_id,
                        'transaction_at': transaction_date + pd.Timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59), seconds=random.randint(0, 59)),
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
                    }, ignore_index=True)
            if data_chunk:
                count_transactions_portfolio += len(data_chunk)
                df.append(data_chunk, ignore_index=True, inplace=True)
        log(f'\tGenerated `{len(count_transactions_portfolio)}` transactions'
            f'for portfolio `{portfolio_id}` (officer `{officer_id}`: `{officer_name}`).',
            end='\n')
    log(f'\nGenerated `{len(df)}` transactions in total.')

    # Create sorted index
    df.set_index(['transaction_year', 'transaction_month', 'transaction_day', 'transaction_id'], inplace=True, drop=False)
    df.sort_index(ascending=True, inplace=True)

    # Save data to disk as Avro partitioned by year
    avro_schema = parse_schema(pyspark_schema)
    for group_name, group_df in df.groupby(level=0):
        path = os.path.join('data', 'card_transactions', f'ptt_transaction_year={group_name}', 'data.avro')
        with open(path, 'wb') as f:
            writer(f, avro_schema, group_df.to_dict(orient='records', index=False))
        log(f'\nSaved `{path}` with `{len(group_df)}` records.')

if __name__ == '__main__':
    generate_dummy_data(
        n_officers=1,
        n_consumers_officer=10,
        n_transactions_consumer_day=6,
        start_date=date(2020, 1, 1),
        end_date=date(2024, 2, 29),
        chaos_consumers_officer=0.5,
        chaos_transactions_client_day=0.5,
        log=print
    )