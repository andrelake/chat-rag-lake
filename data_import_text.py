from env import CHROMA_DB_HOST, CHROMA_DB_PORT, OPENAI_API_KEY
from data_handler.chromadb import (
    log,
    get_chromadb_client,
    get_embeddings_client,
    extract_documents,
    create_collection,
    add_documents,
    query_collection,
    delete_collection,
    show_embeddings_cost,
    get_month_name
)

# Configure Logger
log.verbose = True
log.end = '\n\n'

# Load document
file_path = '*.avro'
data = None

# Chunk data
chunks = None

# Pinecone vectorstore client
pinecone = get_pinecone_client(PINECONE_API_KEY)

# OpenAI embeddings client
embeddings = get_embeddings_client(OPENAI_API_KEY)

# Insert or Fetch Embeddings
index_name = 'ai_prj_prd_data'
if index_name in pinecone.list_indexes():
    delete_pinecone_index(index_name, pinecone)
vectorstore = insert_or_fetch_embeddings(index_name, chunks, pinecone, embeddings)
