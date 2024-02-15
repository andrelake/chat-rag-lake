import os

from dotenv import load_dotenv
from langchain_community.vectorstores import AstraDB
from langchain_openai import OpenAIEmbeddings

api_key = os.getenv('OPENAI_API_KEY')
astradb_token = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
astradb_endpoint = os.getenv('ASTRA_DB_API_ENDPOINT')
collection_name = os.getenv('ASTRA_DB_COLLECTION_NAME')


vectorstore = AstraDB(
        embedding=OpenAIEmbeddings(),
        collection_name=collection_name,
        token=astradb_token,
        api_endpoint=astradb_endpoint
      )

retriever = vectorstore.as_retriever()




if __name__ == '__main__':
    load_dotenv()
