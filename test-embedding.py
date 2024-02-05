import os

from connections import Embeddings
from env import ASTRA_DB_API_ENDPOINT, ASTRA_DB_APPLICATION_TOKEN


class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    CYAN = '\033[96m'
    END = '\033[0m'


def reset_collection(collection_name: str):
    collection = Embeddings.connect_db(
        astra_db_application_token=ASTRA_DB_APPLICATION_TOKEN,
        astra_db_api_endpoint=ASTRA_DB_API_ENDPOINT,
        collection_name=collection_name,
    )

    # Import documents
    json_path = os.path.join('data', 'documents', f'{collection_name}_default.json')
    Embeddings.import_documents(
        collection=collection,
        json_path=json_path
    )
    print(f'{Color.GREEN}Default documents data imported to {Color.CYAN}{collection_name}{Color.GREEN} successfully{Color.END}')

    # Count documents
    # documents_count = collection.count_documents()['status']['count']
    # print(f'Collection {Color.CYAN}{collection_name}{Color.END} has {Color.CYAN}{documents_count}{Color.END} documents')

    # Get documents
    # documents = collection.get()
    # print(f'Collection {Color.CYAN}{collection_name}{Color.END} documents:')
    # print(documents)

    return collection


db_invoices = reset_collection('db_invoices_embed')
