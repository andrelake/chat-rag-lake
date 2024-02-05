import os

from connections import Embeddings
from env import ASTRA_DB_API_ENDPOINT, ASTRA_DB_APPLICATION_TOKEN

import json
from pprint import pprint


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

    # Clear collection
    collection.clear()

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

def test_search(collection):
    with open(os.path.join('data', 'search_vector.json'), 'r') as file:
        vector = json.load(file)
    with open(os.path.join('data', 'search_query.txt'), 'r') as file:
        query = file.read()
    pprint(collection.similarity_search(query, k=3))
    pprint(collection.similarity_search_by_vector(vector, k=3))


db_invoices = reset_collection('db_invoices_embed')
test_search(collection=db_invoices)


# search(query, search_type, **kwargs) Return docs most similar to query using specified search type.
# similarity_search(query[, k, filter]) Return docs most similar to query.
# similarity_search_by_vector(embedding[, k, ...]) Return docs most similar to embedding vector.
# similarity_search_with_relevance_scores(query) Return docs and relevance scores in the range [0, 1].
# similarity_search_with_score(query[, k, filter]) Run similarity search with distance
# similarity_search_with_score_by_vector(embedding) Return docs most similar to embedding vector.
# similarity_search_with_score_id(query[, k, ...])
# similarity_search_with_score_id_by_vector(...) Return docs most similar to embedding vector.