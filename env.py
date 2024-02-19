from os import getenv

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = getenv('OPENAI_API_KEY')
OPENAI_MODEL_NAME = getenv('OPENAI_MODEL_NAME')

PINECONE_API_KEY = getenv('PINECONE_API_KEY')
PINECONE_REGION = getenv('PINECONE_REGION')
PINECONE_INDEX_NAME = getenv('PINECONE_INDEX_NAME')
