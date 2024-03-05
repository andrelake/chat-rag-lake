import os
from typing import Optional, List
from datetime import date
import pytz
import random
from faker import Faker
import pandas as pd
import numpy as np

from utils import Logger, log
from data_handler import read_orc, write_orc


class CardTransactions:
    path = os.path.join('data', 'landing', 'card_transactions.orc')
    schema = {
        'pandas': {
            'transaction_id': np.uint32,
            'transaction_at': pd.StringDtype(),  # pd.DatetimeTZDtype(tz='UTC'),
            'transaction_year': np.uint16,
            'transaction_month': np.uint8,
            'transaction_day': np.uint8,
            'consumer_id_hash': np.uint32,
            'consumer_id': np.uint32,
            'consumer_document': pd.StringDtype(),
            'consumer_name': pd.StringDtype(),
            'portfolio_id': np.uint32,
            'officer_id': np.uint32,
            'officer_name': pd.StringDtype(),
            'product': pd.CategoricalDtype(categories=['', 'credit', 'debit'], ordered=True),
            'card_variant': pd.CategoricalDtype(categories=['', 'black', 'gold', 'platinum', 'standard'], ordered=True),
            'transaction_value': np.float64,
            'seller_description': pd.StringDtype()
        },
        'pyspark': {
            'transaction_id': 'bigint',
            'transaction_at': 'timestamp',
            'transaction_year': 'smallint',
            'transaction_month': 'byte',
            'transaction_day': 'byte',
            'consumer_id_hash': 'bigint',
            'consumer_id': 'bigint',
            'consumer_document': 'string',
            'consumer_name': 'string',
            'portfolio_id': 'bigint',
            'officer_id': 'bigint',
            'officer_name': 'string',
            'product': 'string',
            'card_variant': 'string',
            'transaction_value': 'double',
            'seller_description': 'string'
        }
    }
    metadata = {
        'table_description': 'Card transactions made by consumers.',
        'columns_description': {
            'transaction_id': 'Unique transaction identifier.',
            'transaction_at': 'Date and time of the transaction formatted as "YYYY-MM-DDTHH:MM:SS.sssZ".',
            'transaction_year': 'Year of the transaction.',
            'transaction_month': 'Month of the transaction.',
            'transaction_day': 'Day of the transaction.',
            'consumer_id_hash': 'Unique consumer identifier hash (anonymized).',
            'consumer_id': 'Unique consumer identifier (anonymized).',
            'consumer_document': 'Brazilian 11-digit CPF identification document number of the consumer. Formatted as "00000000000"',
            'consumer_name': 'Name of the consumer/costumer that made the transaction.',
            'portfolio_id': 'Unique identifier of the portfolio of clients managed by the officer.',
            'officer_id': 'Unique identifier of the officer (bank account manager) managing the portfolio of clients.',
            'officer_name': 'Name of the officer managing the portfolio of clients.',
            'product': 'Type of card used: "credit", "debit".',
            'card_variant': 'Variant of the card used: "black", "gold", "platinum", "standard".',
            'transaction_value': 'Value of the transaction in BRL (Brazilian Real).',
            'seller_description': 'Description of the seller/establishment that received the transaction.'
        }
    }
    @staticmethod
    def generate_dummy_data(
        order_by: List,
        n_officers: int,
        n_consumers_officer: int,
        n_transactions_consumer_day: int,
        start_date: date,
        end_date: date,
        chaos_consumers_officer: float = 0,
        chaos_transactions_client_day: float = 0,
        log: Optional[Logger] = log
    ):
        pandas_schema = CardTransactions.schema['pandas']

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
            for consumer_id in range(int(n_consumers_officer * max(1 + random.uniform(-chaos_consumers_officer, chaos_consumers_officer), 0))):
                consumer_document = fake.cpf().replace('-', '').replace('.', '').rjust(11, '0')
                consumer_name = fake.name()
                for transaction_date in pd.date_range(start=start_date, end=end_date, freq='D'):
                    for _ in range(int(n_transactions_consumer_day * max(1 + random.uniform(-chaos_transactions_client_day, chaos_transactions_client_day), 0))):
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
                            'seller_description': fake.company().strip()
                        })
                if data_chunk:
                    count_transactions_portfolio += len(data_chunk)
                    df = pd.DataFrame(data=data_chunk, columns=pandas_schema.keys()) \
                        .astype(pandas_schema)
                    data_chunk = []
                    dfs.append(df)
            log(f'\tGenerated `{count_transactions_portfolio}` transactions '
                f'for portfolio `{portfolio_id}` (officer `{officer_id}`: `{officer_name}`).',
                end='\n')
        
        df = pd.concat(dfs, ignore_index=True)
        
        log(f'\nGenerated `{len(df)}` transactions in total.')
        if order_by:
            df = df.sort_values(order_by)
        df.reset_index(drop=True, inplace=True)

        # Dataframe info, size and memory usage
        log(f'Dataframe info:')
        log(df.info(), end='\n')
        log(f'Dataframe size: `{df.shape}`', end='\n')
        log(f'Dataframe memory usage: `{df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB`')
        
        return df

    def quiz(df: pd.DataFrame, log: Optional[Logger] = log):
        def ask(message):
            log(f'\033[96mPrompt: {message}\033[0m', end='\n')
        
        def answer(message):
            if '\n' in message:
                message = '\n\t' + message.replace('\n', '\n\t')
            log(f'\033[96mAnswer: \033[0m{message}', end='\n\n')

        df2 = df.copy()
        hypothetical_officer_id = df2.officer_id.sample(1).iloc[0]
        hypothetical_officer_name = df2[df2.officer_id == hypothetical_officer_id].officer_name.iloc[0]
        hypothetical_current_month = df2.transaction_month.max()
        hypothetical_current_year = df2.transaction_year.max()


        # Query: Qual o valor total de transações em janeiro de 2023 feitas pelos clientes da carteira do gerente?
        ask(f'Query: Qual o valor total de transações em janeiro de 2023 feitas pelos clientes da carteira do gerente {hypothetical_officer_name}?')
        df2 = df.copy()
        result = df2[(df2.transaction_year == 2023) & (df2.transaction_month == 1) & (df2.officer_id == hypothetical_officer_id)].transaction_value.sum()
        answer(f'R$ {result:.2f}')


        # Query: Quantos clientes possuem um cartão de crédito e quantos possuem um de débito?
        ask(f'Query: Quantos clientes possuem um cartão de crédito e quantos possuem um de débito?')
        df2 = df.copy()
        result = '\n'.join(
            df2[['product', 'consumer_id']] \
               .groupby(['product']) \
               .agg({'consumer_id': 'nunique'}) \
               .apply(lambda record: f'{record.name}: {record.consumer_id} clientes', axis=1) \
               .to_list()
        )
        answer(result)


        # Query: Quantos clientes realizaram mais de R$ 8000 em transações com cartão platinum em um único mês?
        ask(f'Query: Quantos clientes realizaram mais de R$ 8000 em transações com cartão platinum em um único mês?')
        df2 = df.copy()
        result = df2[df2.card_variant == 'platinum'].groupby(['transaction_year', 'transaction_month', 'consumer_id']).transaction_value.sum().gt(8000).sum()
        answer(f'{result} clientes')


        # Query: Quantas transações foram realizadas nas 3 carteiras com o maior
        ask(f'Query: Quantas transações foram realizadas pelos 3 clientes com o maior valor total de transações em abril de 2023?')
        df2 = df.copy()
        consumers = df2[df2.officer_id == hypothetical_officer_id].groupby('consumer_id').transaction_value.sum().nlargest(3).index
        result = df2[(df2.transaction_year == 2023) & (df2.transaction_month == 4) & (df2.consumer_id.isin(consumers))].shape[0]
        answer(f'{result} transações.')


        # Quem são os meus 10 clientes que mais gastam no crédito?
        # Resposta: Lista com consumer_id, consumer_name, sum_transaction_value de cada cliente
        ask('Quem são os 10 clientes que mais gastam no crédito?')
        ask(f'...Eu me chamo {hypothetical_officer_name}.')
        df2 = df.copy()
        result = '\n'.join(
            df2[(df2['product'] == 'credit') & (df2.officer_id == hypothetical_officer_id)] \
               .groupby(['consumer_id', 'consumer_name']) \
               .agg({'transaction_value': 'sum'}) \
               .sort_values('transaction_value', ascending=False) \
               .reset_index() \
               .iloc[:10] \
               .apply((lambda record: f'[ID {record.consumer_id:09}] {record.consumer_name}  >  R$ {record.transaction_value:.2f}'), axis=1) \
               .to_list()
        )
        answer(result)


        # Qual é o total de gastos feito no cartão dos clientes da minha carteira?
        # Resposta: Valor total de gastos
        ask('Qual é o total de gastos feito no cartão dos clientes da minha carteira?')
        ask(f'...Eu me chamo {hypothetical_officer_name}.')
        df2 = df.copy()
        result = df2[df2.officer_id == hypothetical_officer_id].transaction_value.sum()
        answer(f'{result:.2f}')


        # Mostre o ranking entre gastos de cartão dos clientes por team leader
        # Fora do escopo de carteira por gerente


        # Mostre a evolução de gastos dos últimos 4 meses para 3 clientes da carteira
        # Resposta:
        ask('Mostre a evolução de gastos dos últimos 4 meses para 3 clientes da carteira')
        df2 = df.copy()
        hypothetical_consumer_ids = df2.consumer_id.sample(3).unique()
        result = df2[(df2.consumer_id.isin(hypothetical_consumer_ids)) & (df2.transaction_month >= hypothetical_current_month - 4) & (df2.transaction_year == hypothetical_current_year)] \
                    .groupby(['consumer_id', 'consumer_name', 'transaction_year', 'transaction_month']) \
                    .transaction_value.sum() \
                    .reset_index() \
                    .sort_values(['consumer_id', 'transaction_year', 'transaction_month'])
        result['_agg'] = result.apply(lambda record: f'\t> {record.transaction_month:02}/{record.transaction_year}: R$ {record.transaction_value:.2f}', axis=1)
        result = '\n'.join(
            result.groupby(['consumer_id', 'consumer_name']) \
                  .agg({'_agg': '\n'.join}) \
                  .reset_index() \
                  .apply(lambda record: f'{record.consumer_name} ({record.consumer_id}):\n{record._agg}', axis=1) \
                  .to_list()
        )
        answer(result)

        # Mostre o ranking de gastos de cartão dos clientes por estabelecimento
        # Resposta:
        ask('Mostre o ranking de gastos de cartão dos clientes por estabelecimento')
        df2 = df.copy()
        result = '\n'.join(
            df2.groupby(['seller_description']) \
               .transaction_value.sum() \
               .sort_values(ascending=False) \
               .reset_index() \
               .iloc[:5] \
               .apply((lambda record: f'{record.seller_description}: R$ {record.transaction_value:.2f}'), axis=1) \
               .to_list()
        )
        answer(result)

        # Liste quem são os clientes que não possui utilização de cartão
        # Fora do escopo de transações


        # Liste quem são os clientes com gastos mensais abaixo de R$ 3000
        # Resposta:
        ask('Liste quem são os clientes com gastos mensais abaixo de R$ 3000')
        df2 = df.copy()
        result = '\n'.join(
            df2.groupby(['consumer_id', 'consumer_name', 'transaction_year', 'transaction_month']) \
               .transaction_value \
               .sum() \
               .lt(3000) \
               .reset_index() \
               [['consumer_id', 'consumer_name']] \
               .drop_duplicates() \
               .apply((lambda record: f'{record.consumer_name} (ID {record.consumer_id:09})'), axis=1) \
               .to_list()
        )
        answer(result)

        # Liste os clientes das carteiras que vem reduzindo os gastos com cartão ?
        # Resposta:
        ask('Liste os clientes das carteiras que vem reduzindo os gastos com cartão nos últimos 4 meses')
        df2 = df.copy()
        months_ordered = df2[['transaction_year', 'transaction_month']] \
                            .drop_duplicates() \
                            .sort_values(['transaction_year', 'transaction_month']) \
                            .apply((lambda x: (x.transaction_year, x.transaction_month)), axis=1)
        hypothetical_current_year_minus1month, hypothetical_current_month_minus1month = months_ordered.iloc[-2]
        hypothetical_current_year_minus4month, hypothetical_current_month_minus4month = months_ordered.iloc[-5]
        last_periods = df2.groupby(by=['consumer_id', 'consumer_name', 'transaction_year', 'transaction_month']) \
                 .transaction_value \
                 .sum() \
                 .reset_index() \
                 .sort_values(by=['consumer_id', 'transaction_year', 'transaction_month'])
        last_periods = last_periods[
            (last_periods.apply((lambda x: f'{x.transaction_year:04}-{x.transaction_month:02}'), axis=1) < f'{hypothetical_current_year:04}-{hypothetical_current_month:02}') &
            (last_periods.transaction_year >= hypothetical_current_year_minus4month) &
            (last_periods.transaction_month >= hypothetical_current_month_minus4month)
        ]
        last_period = df2[(df2.transaction_year == hypothetical_current_year_minus1month) & (df2.transaction_month == hypothetical_current_month_minus1month)] \
                         .groupby(['consumer_id', 'consumer_name']) \
                         .agg({'transaction_value': 'sum'}) \
                         .sort_index()
        reducing_consumers = last_period[
            last_period.transaction_value <
                last_periods.groupby(['consumer_id', 'consumer_name']) \
                            .agg({'transaction_value': 'mean'}) \
                            .sort_index() \
                            .transaction_value
        ]
        last_periods = last_periods[last_periods.consumer_id.isin(reducing_consumers.reset_index().consumer_id)]
        last_periods['_agg'] = last_periods.apply((lambda record: f'\t> {record.transaction_month:02}/{record.transaction_year}: R$ {record.transaction_value:.2f}'), axis=1)
        last_periods = last_periods.groupby(['consumer_id', 'consumer_name']).agg({'_agg': '\n'.join}).reset_index()
        last_periods['_agg2'] = last_periods.apply((lambda record: f'{record.consumer_name} (ID {record.consumer_id:09}):\n{record._agg}'), axis=1)
        result = '\n'.join(last_periods._agg2.to_list())
        answer(result)
        del df2, last_periods, last_period, reducing_consumers, months_ordered, hypothetical_current_year_minus1month, hypothetical_current_month_minus1month, hypothetical_current_year_minus4month, hypothetical_current_month_minus4month

        # Liste os clientes que possuem gastos nos últimos 6 meses para realizarmos upgrade de variante e não pagar mensalidade?
        # Liste os clientes que estão próximos da utilização total do limite, bons pagadores que poderíamos majorar o LMC de acordo com bacen?
        # Liste quem são os clientes com LMC pré-aprovado de cartões do maior valor para o menor
        # Liste quem são os clientes com atraso em fatura de cartões
        # Liste quem são os clientes com LMC aprovado e não contratado do maior para o menor valor
        # Liste quem são os clientes com gastos mensais superior a R$ 8.0000
        # Liste quem são os clientes com gastos mensais superior a R$ 3.0000 até 7.999,00
        # Liste quem são os últimos clientes da carteira que contrataram cartão de crédito
        # Liste quem são os clientes com pendencia de analise nos documentos para contratação do cartão de crédito
        # Qual a margem de receita gerada pelos clientes com utilização de cartão da minha carteira
        # Liste clientes que possui portabilidade de salário ativo e não possui a variante de cartão Platinum e/ou Black contratado.
        # Liste clientes que possui investimentos a partir de 20 mil reais e não possui a variante de cartão Platinum contratado.
        # Liste clientes que possui investimentos a partir de 50 mil reais e não possui a variante de cartão Black contratado.
        # Quais os clientes que possuem gastos mensais médios no valor de 3.000, e ainda não possuem a variante platinum?
        # Quais os clientes que possuem gastos mensais médios no valor de 8.000, e ainda não possuem a variante black?
        # Quais os clientes com mais de 50k em investimentos sem cartão ativo?
        # Quais os clientes com portabilidade ativa, e cartão pre aprovado?
        # Mostre os clientes com cartões de crédito de titularidade própria, de terceiros(instituições) cadastrados na carteira e pré aprovado com PicPay

        # Perguntas adicionais
        # Quantos % dos clientes da carteira do gerente # fizeram transações com cartão de crédito nos últimos 6 meses?

    def read() -> pd.DataFrame:
        schema = CardTransactions.schema['pandas']
        df = read_orc(path=CardTransactions.path, log=log).astype(schema)
        return df
    
    def write(df: pd.DataFrame):
        write_orc(
            df=df,
            path=CardTransactions.path,
            partitionBy=['transaction_year'],
            compression='zstd',
            log=log
        )


if __name__ == '__main__':
    log.verbose = True
    log.end = '\n\n'
    df = CardTransactions.generate_dummy_data(
        order_by=[
            'transaction_year',
            'portfolio_id',
            'consumer_id',
            'transaction_at',
        ],
        n_officers=1,
        n_consumers_officer=10,
        n_transactions_consumer_day=3,
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
        chaos_consumers_officer=0,
        chaos_transactions_client_day=0.66,
        log=log
    )
    CardTransactions.quiz(df, log)