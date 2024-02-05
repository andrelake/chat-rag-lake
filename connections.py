import json

from astrapy.db import AstraDB, AstraDBCollection
from env import OPENAI_API_KEY, ASTRA_DB_API_ENDPOINT, ASTRA_DB_APPLICATION_TOKEN

from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

from typing import Dict, Optional, List


class Default:
    def connect_db(astra_db_application_token: str, astra_db_api_endpoint: str, namespace: str = 'default_keyspace'):
        return AstraDB(
            token=astra_db_application_token,
            api_endpoint=astra_db_api_endpoint,
            namespace=namespace
        )

    def import_documents(collection: AstraDBCollection, json_path: str):
        with open(json_path, 'r') as file:
            documents = json.load(file)
        collection.insert_many(documents)

    db = connect_db(
        astra_db_application_token=ASTRA_DB_APPLICATION_TOKEN,
        astra_db_api_endpoint=ASTRA_DB_API_ENDPOINT,
    )


class Embeddings:
    # Not working
    def connect_db(openai_api_key: str, astra_db_application_token: str, astra_db_api_endpoint: str, collection_name: str):
        from langchain_community.vectorstores import AstraDB as LangchainAstraDB


        embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        collection = LangchainAstraDB(
            embedding=embedding,
            collection_name=collection_name,
            token=astra_db_application_token,
            api_endpoint=astra_db_api_endpoint,
        )
        return collection

    def import_documents(collection: AstraDBCollection, json_path: str):
        with open(json_path, 'r') as file:
            documents = json.load(file)

        # Add a LangChain document with the quote and metadata tags
        documents = [
            Document(
                page_content=json.dumps(document, separators=('\t', '\t')).replace('{', '')
                                                                          .replace('}', '')
                                                                          .replace('"', ''),
                metadata={'teste': 'teste'}
            )
            for document in documents
        ]
        collection.add_documents(documents)
    
    def similarity_search(collection, query: str, k: int = 4, filter: Optional[Dict[str, str]] = None) -> List[Document]:
        return collection.similarity_search(query, k, filter)
    
    def similarity_search_by_vector(collection, embedding: List[float], k: int = 4, filter: Optional[Dict[str, str]] = None) -> List[Document]:
        return collection.similarity_search_by_vector(embedding, k, filter)

