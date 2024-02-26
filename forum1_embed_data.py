import os
from pprint import pprint

from env import PINECONE_API_KEY, OPENAI_API_KEY, PINECONE_ENVIRONMENT
from utils import log
from connections.pinecone import (
    get_database_client,
    create_vectorstore,
    get_vectorstore,
    delete_vectorstore,
    add_documents,
    query_documents,
)
from connections.openai import get_embeddings_client, get_embedding_cost
from data_handler.pdf import load_documents, redistribute_documents


# Configure Logger
log.verbose = False
log.end = '\n\n'

# Load documents
file_path = os.path.join('data', 'landing', 'forum_estatais_educacao.pdf', 'texto.pdf')
data = load_documents(file_path)

# Chunk data
documents = redistribute_documents(data, chunk_size=1600, chunk_overlap=160)

log.verbose = True

# Get vectorstore client
database_client = get_database_client(PINECONE_API_KEY)

# OpenAI embeddings client
embedding_model_name = 'text-embedding-3-small'
embedding_function = get_embeddings_client(OPENAI_API_KEY, model_name=embedding_model_name)

# Get or create vectorstore
vectorstore_name = 'felipe-dev-picpay-prj-ai-rag-llm-pdf'
vectorstore = get_vectorstore(vectorstore_name, embedding_function, database_client, create=True, dimension_count=None)

# Add documents
get_embedding_cost(documents=documents, model_name=embedding_model_name)
# add_documents(vectorstore, documents)