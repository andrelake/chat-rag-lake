import json

from astrapy.db import AstraDB, AstraDBCollection
from env import OPENAI_API_KEY, ASTRA_DB_API_ENDPOINT, ASTRA_DB_APPLICATION_TOKEN

from langchain_openai import OpenAIEmbeddings

def connect(astra_db_application_token: str, astra_db_api_endpoint: str, namespace: str = 'default_keyspace'):
    return AstraDB(
        token=astra_db_application_token,
        api_endpoint=astra_db_api_endpoint,
        namespace=namespace
    )

def import_documents(collection: AstraDBCollection, json_path: str):
    with open(json_path, 'r') as file:
        documents = json.load(file)
    collection.insert_many(documents)

db = connect(
    astra_db_application_token=ASTRA_DB_APPLICATION_TOKEN,
    astra_db_api_endpoint=ASTRA_DB_API_ENDPOINT,
)


# Not working
def connect_with_langchain_openai(astra_db_application_token: str, astra_db_api_endpoint: str, collection_name: str):
    from langchain_community.vectorstores import AstraDB as LangchainAstraDB


    embedding = OpenAIEmbeddings()
    coll_movies_db = LangchainAstraDB(
        embedding=embedding,
        collection_name=collection_name,
        token=astra_db_application_token,
        api_endpoint=astra_db_api_endpoint,
    )
    return coll_movies_db
