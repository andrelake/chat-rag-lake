from utils import log


log.verbose = True
log.end = '\n\n'


from env import PINECONE_API_KEY, OPENAI_API_KEY, PINECONE_INDEX_NAME
from connections.pinecone import get_database_client, get_vectorstore, query_documents
from connections.embeddings import get_client, get_dimension_count


if __name__ == '__main__':
    class Embedding:
        model_name = 'text-embedding-3-large'
        type = 'api'
        client = get_client(model_name=model_name, type=type, api_key=OPENAI_API_KEY)
        dimension_count = get_dimension_count(model_name=model_name, type=type)
    

    class Database:
        client = get_database_client(PINECONE_API_KEY)
        vectorstore_name = PINECONE_INDEX_NAME
                         # 'prj-ai-rag-llm-table-1-standard'
                         # 'prj-ai-rag-llm-table-2-discursive'
                         # 'prj-ai-rag-llm-table-3-standard-creditcard'
                         # 'prj-ai-rag-llm-table-4-discursive-creditcard'
                         # 'prj-ai-rag-llm-table-5-join-wo-transactions-2'
                         # 'prj-ai-rag-llm-table-6-chunks'
                         # 'prj-ai-rag-llm-table-7-cohere'
                         # 'prj-ai-rag-llm-table-8-storytelling'
                         # 'prj-ai-rag-llm-table-9-questions'
                         # 'prj-ai-rag-llm-table-10-granularity'
                         # 'prj-ai-rag-llm-table-11-large-embedding'
        vectorstore = get_vectorstore(
            name=vectorstore_name,
            embedding_function=Embedding.client,
            database_client=client,
            create=False,
            dimension_count=Embedding.dimension_count
        )


    while True:
        prompt = input("\033[96mEnter a prompt (type 'exit' to quit):\033[0m ")
        if prompt == "exit":
            break

        # Search for similar documents
        result_documents = query_documents(Database.vectorstore, prompt, k=6)
        for document in result_documents:
            # pprint(document.metadata)
            print(document.page_content, end='\n\n')
