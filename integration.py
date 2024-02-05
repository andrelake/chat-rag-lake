import os
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.schema import Document

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

from datasets import load_dataset

from db_conn import coll_movies_db


# set db
vstore = coll_movies_db

# load data:
movies_dataset = load_dataset("")
print("An example entry:")
print(movies_dataset[16])
