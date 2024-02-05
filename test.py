import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(openai_api_key=api_key)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a software developer with a huge knowledge in python, LLM, Vectorial Database and RAG."),
    ("user", "{input}")
])

output_parser = StrOutputParser()

chain = prompt | llm | output_parser
resp = chain.invoke({"input": "how can Langchain help a RAG made with Python and a vectorial database?"})
print(resp)
