from dotenv import load_dotenv
from os import getenv


load_dotenv()

OPENAI_API_KEY = getenv('OPENAI_API_KEY')
ASTRA_DB_APPLICATION_TOKEN = getenv('ASTRA_DB_APPLICATION_TOKEN')
ASTRA_DB_API_ENDPOINT = getenv('ASTRA_DB_API_ENDPOINT')
