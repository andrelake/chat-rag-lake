from typing import Union, Tuple, List, Set
from env import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_REGION

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Pinecone as PineconeVectorstore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
import tiktoken


def get_pinecone_client(api_key: str) -> Pinecone:
    return Pinecone(api_key=api_key)


def get_embeddings_client(api_key: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(openai_api_key=api_key)


def load_document(filepath) -> List:
    print(f'Loading {filepath}')
    loader = PyPDFLoader(filepath)
    data = loader.load()
    print(f'\nData Loaded Successfully. Total pages: {len(data)}')
    return data


def chunk_data(data: List, chunk_size: int = 1600) -> List:
    print(f'\nStarting chunk context')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0
    )
    chunks = text_splitter.split_documents(data)
    i = 1
    for chunk in chunks:
        print(f'#{i} {chunk.page_content}\n')
        i += 1
    print(f'\nTotal chunks: {len(chunks)}')
    return chunks


def insert_or_fetch_embeddings(index_name: str, chunks: List, pinecone: Pinecone, embeddings: OpenAIEmbeddings) -> PineconeVectorstore:
    indexes = pinecone.list_indexes()
    print(f'\nIndexes: {len(indexes)}')
    if len(indexes) == 0:
        return create_vector_store(chunks, embeddings, index_name, pinecone)
    else:
        for index in indexes:
            if index['name'] == index_name:
                print(f'\nGetting info from index: {index_name}')
                return PineconeVectorstore.from_existing_index(index_name, embeddings)
            else:
                return create_vector_store(chunks, embeddings, index_name, pinecone)


def create_vector_store(chunks: List, embeddings: OpenAIEmbeddings, index_name: str, pinecone: Pinecone) -> PineconeVectorstore:
    print(f'\nCreating index: {index_name}')
    pinecone.create_index(
        index_name,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region=PINECONE_REGION)
    )
    return PineconeVectorstore.from_documents(chunks, embeddings, index_name=index_name)


def delete_pinecone_index(index_name: str, pinecone: Pinecone) -> None:
    print(f'\nDeleting index: {index_name}"')
    pinecone.delete_index(index_name)
    print(f'\nIndex Deleted Successfully.')


def delete_all_pinecone_indexes(pinecone: Pinecone) -> None:
    print(f'\nDeleting all indexes...')
    indexes = pinecone.list_indexes()
    for index in pinecone.list_indexes():
        pinecone.delete_index(index)
    print(f'\nSuccessfully Deleted All Indexes: {", ".join(indexes)}.')


def show_embeddings_cost(texts: Union[Tuple, List, Set]) -> None:
    print(f'\nStarting embedding price calculation')
    encoding = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(encoding.encode(page.page_content)) for page in texts])
    print(f'Total tokens: {total_tokens}')
    print(f'Total pages: {len(texts)}')
    print(f'Embedding cost: ${total_tokens * 0.0004 / 1000:.4f}')


if __name__ == '__main__':
    # Load document
    file_path = 'data/texto.pdf'
    data = load_document(file_path)

    # Chunk data
    chunks = chunk_data(data)

    # Pinecone vectorstore client
    pinecone = get_pinecone_client(PINECONE_API_KEY)

    # OpenAI embeddings client
    embeddings = get_embeddings_client(OPENAI_API_KEY)

    # Insert or Fetch Embeddings
    index_name = 'felipe-test-index-1'
    vectorstore = insert_or_fetch_embeddings(index_name, chunks, pinecone, embeddings)

    # Show Embeddings Cost
    show_embeddings_cost(chunks)

    # Delete Pinecone Index
    # delete_pinecone_index(index_name, pinecone)

    # Delete All Pinecone Indexes
    # delete_all_pinecone_indexes(index_name, pinecone)
