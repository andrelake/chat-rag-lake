import os

from connections import db, import_documents


class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    CYAN = '\033[96m'
    END = '\033[0m'


def reset_collection(collection_name: str):
    # Delete collection if exists
    if collection_name in db.get_collections()['status']['collections']:
        db.delete_collection(collection_name=collection_name)
        print(f'{Color.GREEN}Existing collection {Color.CYAN}{collection_name}{Color.GREEN} deleted successfully{Color.END}')
        
    # Create a collection
    db.create_collection(
        collection_name=collection_name,
        dimension=2
    )
    collection = db.collection(collection_name)
    print(f'{Color.GREEN}Collection {Color.CYAN}{collection_name}{Color.GREEN} created successfully{Color.END}')


    # Import documents
    json_path = os.path.join('data', 'documents', f'{collection_name}_default.json')
    import_documents(
        collection=collection,
        json_path=json_path
    )
    print(f'{Color.GREEN}Default documents data imported to {Color.CYAN}{collection_name}{Color.GREEN} successfully{Color.END}')

    # Count documents
    documents_count = collection.count_documents()['status']['count']
    print(f'Collection {Color.CYAN}{collection_name}{Color.END} has {Color.CYAN}{documents_count}{Color.END} documents')

    return collection


db_invoices = reset_collection('db_invoices')
