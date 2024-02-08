import json

from astrapy.db import AstraDB, AstraDBCollection

import sqlite3

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


class SQLite:
    ddl = {
        'portfolio_invoices': '''
            CREATE TABLE IF NOT EXISTS invoices (
                id INTEGER PRIMARY KEY,
                date TEXT,
                description TEXT,
                type TEXT,
                value REAL,
                consumer_name TEXT,
                consumer_cpf TEXT
            );
        '''
    }
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.conn = sqlite3.connect(self.file_path)
        self.cur = self.conn.cursor()
    
    def create_table(self, table_name: str):
        self.cur.execute(self.ddl[table_name])
        self.conn.commit()
    
    def clear_table(self, table_name: str):
        self.cur.execute(f'DELETE FROM {table_name};')
        self.conn.commit()
    
    def import_csv(self, file_path: str, table_name: str, sep: str = ','):
        with open(file_path, 'r') as file:
            data = file.readlines()
        data = (line.split(sep) for line in data)
        sql = f'''
            INSERT INTO t VALUES ({', '.join('?' * len(data[0]))});
        '''
        self.cur.executemany(sql, data)
        self.conn.commit()
    
    def terraform_db(self, file_path: str):
        # List tables
        sql_list_tables = '''
            SELECT name FROM sqlite_master WHERE type='table';
        '''
        tables = self.cur.execute(sql_list_tables).fetchall()

        # Create tables
        for table in self.ddl:
            if table in tables:
                self.clear_table(table)
            else:
                self.create_table(table)
            self.import_csv(self.file_path, table)
