import os
from langchain_community.vectorstores import AstraDB
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.schema import Document

from langchain_openai import OpenAIEmbeddings

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

from datasets import load_dataset

from env import OPENAI_API_KEY, ASTRA_DB_API_ENDPOINT, ASTRA_DB_APPLICATION_TOKEN

# set embedding model
embedding = OpenAIEmbeddings()

# set db
vstore = AstraDB(
    embedding=embedding,
    collection_name="movies_db",
    token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
    api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
)

# load data:
movies_dataset = load_dataset("")
print("An example entry:")
print(movies_dataset[16])