from env import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_REGION


def load_document(filepath) -> list:
    from langchain_community.document_loaders import PyPDFLoader
    print(f"Loading {filepath}")

    loader = PyPDFLoader(filepath)
    data = loader.load()

    print(f"\nData Loaded Successfully. Total pages: {len(data)}")
    return data


def chunk_data(data: list, chunk_size=1600) -> list:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    print(f"\nStarting chunk context")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0
    )
    chunks = text_splitter.split_documents(data)
    i = 1
    for chunk in chunks:
        print(f"#{i} {chunk.page_content}\n")
        i += 1

    print(f"\nTotal chunks: {len(chunks)}")
    return chunks


def insert_or_fetch_embeddings(index_name: str, chunks: list):
    from pinecone import Pinecone
    from langchain_community.vectorstores import Pinecone as PineconeVectorstore
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    pinecone = Pinecone(
        api_key=PINECONE_API_KEY
    )

    indexes = pinecone.list_indexes()
    print(f"\nIndexes: {len(indexes)}")

    if len(indexes) == 0:
        return create_vector_store(chunks, embeddings, index_name, pinecone)
    else:
        for index in indexes:
            if index['name'] == index_name:
                print(f"\nGetting info from index: {index_name}")
                return PineconeVectorstore.from_existing_index(index_name, embeddings)
            else:
                return create_vector_store(chunks, embeddings, index_name, pinecone)


def create_vector_store(chunks, embeddings, index_name, pinecone):
    from langchain_community.vectorstores import Pinecone as PineconeVectorstore
    from pinecone import ServerlessSpec
    print(f"\nCreating index: {index_name}")
    pinecone.create_index(index_name, dimension=1536, metric="cosine", spec=ServerlessSpec(
        cloud='aws',
        region=PINECONE_REGION,
    ))
    return PineconeVectorstore.from_documents(chunks, embeddings, index_name=index_name)


def delete_pinecone_index(index_name='all'):
    from pinecone import Pinecone

    pinecone = Pinecone(
        api_key=PINECONE_API_KEY
    )

    if index_name == 'all':
        for index in pinecone.list_indexes():
            print(f"\nDeleting all indexes...")
            pinecone.delete_index(index)
            print("\nAll Indexes Deleted Successfully.")
    else:
        print(f"\nDeleting index: {index_name}", end="")
        pinecone.delete_index(index_name)
        print("\nIndex Deleted Successfully.")


def show_embeddings_cost(texts):
    import tiktoken
    print(f"\nStarting embedding price calculation")
    encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
    total_tokens = sum([len(encoding.encode(page.page_content)) for page in texts])
    print(f"Total tokens: {total_tokens}")
    print(f"Total pages: {len(texts)}")
    print(f"Embedding cost: ${total_tokens * 0.0004 / 1000:.4f}")
