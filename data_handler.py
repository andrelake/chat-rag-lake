import os
from shutil import rmtree
from typing import Optional, Any, Callable, List
import pytz
import random
from datetime import date

import pandas as pd
from faker import Faker
from fastavro import block_reader, parse_schema, writer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


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
    df.set_index(group_by, inplace=True, drop=True)
    df.sort_index(ascending=True, inplace=True)

    # Dataframe info, size and memory usage
    log(f'Dataframe info:')
    log(df.info(), end='\n')
    log(f'Dataframe size: `{df.shape}`', end='\n')
    log(f'Dataframe memory usage: `{df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB`')

    if save_path:
        # Delete old data
        log(f'Deleting old data from `{save_path}`...')
        rmtree(save_path, ignore_errors=True)

        # Save data to disk as Avro partitioned by year
        for group_name, group_df in df.groupby(level=0):
            path = os.path.join(save_path, f'ptt_transaction_year={group_name}', 'data.avro')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                writer(f, avro_schema, group_df.reset_index(drop=False).to_dict(orient='records'))
            log(f'Saved `{path}` with `{len(group_df)}` records.', end='\n')
    
    return df


def validation_quiz(df: pd.DataFrame, log: callable = log):
    officer_name = df.iloc[random.randint(0, len(df))].officer_name
    log(f'Query: Qual o valor total de transações em janeiro de 2023 feitas pelos clientes da carteira do gerente {officer_name}?', end='\n')
    result = df[(df.index.get_level_values('transaction_year') == 2023) & (df.transaction_month == 1) & (df.officer_name == officer_name)].transaction_value.sum()
    log(f'Correct answer: `{result}`.')

    log(f'Query: Quantos clientes possuem cartão de crédito e quantos de débito?', end='\n')
    result = df.card_variant.value_counts()
    log(f'Correct answer: `{result}`.')

    log(f'Query: Quantos clientes realizaram mais de R$ 8000 em transações com cartão platinum em um único mês?', end='\n')
    result = df[df.card_variant == 'platinum'].groupby(['transaction_year', 'transaction_month', 'consumer_id']).transaction_value.sum().gt(8000).sum()
    log(f'Correct answer: `{result}`.')

    log(f'Query: Quantas transações foram realizadas nas 3 carteiras com o maior valor total de transações em abril de 2023?', end='\n')
    result = df[(df.index.get_level_values('transaction_year') == 2023) & (df.transaction_month == 4)].groupby('portfolio_id').transaction_value.sum().nlargest(3)
    log(f'Correct answer: `{result}`.')


def get_month_name(n: int) -> str:
    return ('janeiro','fevereiro','março','abril','maio','junho','julho','agosto','setembro','outubro','novembro','dezembro')[n-1]


def extract_documents_from_file(
    file_path: str,
    group_by: str,
    group_body: Callable[[Any], str],
    aggregated_body: Callable[[Any], str],
    filter: Optional[Callable[[Any], bool]] = None
) -> List[Document]:  # TODO: Test pandas implementation
    document: Document = None
    documents: List[Document] = list()
    stream_last_group_headers = None
    stream_current_group_headers = None
    count_processed_rows: int = 0
    count_total_rows: int = 0

    log(f'Reading data from `{file_path}`')
    with open(file_path, 'rb') as fp:
        for block in block_reader(fp):
            count_total_rows += block.num_records
            try:
                for record in block:
                    if not filter(record):
                        continue
                    stream_current_group_headers = tuple(record[header] for header in group_by)
                    if stream_current_group_headers != stream_last_group_headers:
                        if document:
                            documents.append(document)
                        document = Document(
                            page_content=group_body(record),
                            metadata=dict(record)
                        )
                        stream_last_group_headers = stream_current_group_headers
                    document.page_content += aggregated_body(record)
                    count_processed_rows += 1
            except Exception as e:
                log(f'Error: {e}. Skipping record.', end='\n')
                continue
        if document:
            documents.append(document)
    log(f'Processed {count_processed_rows:_}/{count_total_rows:_} rows from `{file_path}` into {len(documents)} documents.')
    return documents


def extract_documents(
    path: str,
    group_by: str,
    group_body: Callable[[Any], str],
    aggregated_body: Callable[[Any], str],
    filter: Optional[Callable[[Any], bool]] = None
) -> List[Document]:  # TODO: Test pandas implementation
    
    log(f'Extracting documents from `{path}`')
    source_files = []
    # Example: filepath = 'data/card_transactions/'
    # Real file: 'data/card_transactions/transaction_year=2023/part-00002-tid-1056622170898768028-a56bd3a2-d283-4b7a-ab87-62442d905a78-116499-1.c000.avro'
    # Example: files = ['data/card_transactions/transaction_year=2023/part-00002-tid-1056622170898768028-a56bd3a2-d283-4b7a-ab87-62442d905a78-116499-1.c000.avro']
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.avro'):
                file_path = os.path.join(root, file)
                source_files.append(file_path)
                log(file_path, end='\n')

    documents = []
    for file_path in source_files:
        documents.extend(
            extract_documents_from_file(
                file_path=file_path,
                group_by=group_by,
                group_body=group_body,
                aggregated_body=aggregated_body,
                filter=filter
            )
        )
    return documents


def load_documents_from_pdf(filepath: str) -> List:
    log(f'Loading {filepath}')
    loader = PyPDFLoader(filepath)
    data = loader.load()
    for page in data:
        page.page_content = re.sub(r' +', ' ', page.page_content)
        page.page_content = re.sub(r'(?: \n)+(?<! )', ' \n', page.page_content)
    log(f'Data Loaded Successfully. Total pages: {len(data)}')
    return data


def redistribute_chunks(data: List, chunk_size: int = 1600, chunk_overlap: int = 0) -> List:
    log(f'Starting chunk context')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(data)
    for i, chunk in enumerate(chunks):
        log(f'#{i+1} {chunk.page_content}')
    log(f'Total chunks: {len(chunks)}')
    return chunks


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

    validation_quiz(df, log)

