import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from astrapy.db import AstraDB

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
db_token = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
db_api_endpoint = os.getenv('ASTRA_DB_API_ENDPOINT')

# llm = ChatOpenAI(openai_api_key=api_key)
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a software developer with a huge knowledge in python, LLM, Vectorial Database and RAG."),
#     ("user", "{input}")
# ])


# Initialization
db = AstraDB(
  token=db_token,
  api_endpoint=db_api_endpoint,
  namespace="default_keyspace")

print(f"Connected to Astra DB: {db.get_collections()}")
print(f"Connected to Astra DB: {db.collection('movies_db')}")
print(f"Connected to Astra DB: {db.collection('movies_db').get()}")

# output_parser = StrOutputParser()
#
# chain = prompt | llm | output_parser
# resp = chain.invoke({"input": "how can Langchain help a RAG made with Python and a vectorial database?"})
# print(resp)
