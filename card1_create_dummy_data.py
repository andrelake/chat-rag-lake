import os
from datetime import date

from utils import log
from data_handler.avro import (
    generate_dummy_data,
    extract_documents,
    validation_quiz
)


# Configure Logger
log.verbose = True
log.end = '\n\n'

avro_path = os.path.join('data', 'landing', 'card_transactions.avro')

# Generate dummy data
df = generate_dummy_data(
    order_by=[
        'transaction_year',
        'portfolio_id',
        'consumer_id',
        'transaction_at',
    ],
    n_officers=1,
    n_consumers_officer=5,
    n_transactions_consumer_day=2,
    start_date=date(2023, 1, 1),
    end_date=date(2024, 2, 29),
    chaos_consumers_officer=0,
    chaos_transactions_client_day=1,
    log=log,
    save_path=avro_path
)
validation_quiz(df, log)
