import os
from datetime import date

from utils import log
from data_tables import CardTransactions


# Configure Logger
log.verbose = True
log.end = '\n\n'

if __name__ == '__main__':
    # Generate dummy data
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
    CardTransactions.quiz(df, save=True)
    CardTransactions.write(df)
