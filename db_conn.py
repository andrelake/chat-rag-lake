from langchain_community.vectorstores import AstraDB as LangchainAstraDB
from astrapy.db import AstraDB as AstrapyAstraDB
from env import OPENAI_API_KEY, ASTRA_DB_API_ENDPOINT, ASTRA_DB_APPLICATION_TOKEN

from langchain_openai import OpenAIEmbeddings


db = AstrapyAstraDB(
    token=ASTRA_DB_APPLICATION_TOKEN,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    namespace="default_keyspace"
)

embedding = OpenAIEmbeddings()

coll_movies_db = LangchainAstraDB(
    embedding=embedding,
    collection_name="movies_db",
    token=ASTRA_DB_APPLICATION_TOKEN,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
)
