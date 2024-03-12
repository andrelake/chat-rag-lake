from pprint import pprint
import json

from utils import log


log.verbose = True
log.end = '\n\n'


from env import PINECONE_API_KEY, OPENAI_API_KEY
from connections.openai import get_self_query_retriever
from connections.pinecone import get_database_client, get_vectorstore, query_documents
from data_tables import CardTransactions
from data_handler import get_embeddings_client


if __name__ == '__main__':
    # Get database client
    database_client = get_database_client(PINECONE_API_KEY)

    # Get embeddings client
    embedding_model_name = 'text-embedding-3-small'
    embedding_function = get_embeddings_client(model_name=embedding_model_name, type='api', api_key=OPENAI_API_KEY)

    # Get vectorstore
    vectorstore_name = (
        'prj-ai-rag-llm-table-1-standard',
        'prj-ai-rag-llm-table-2-discursive',
        'prj-ai-rag-llm-table-3-standard-creditcard',
        'prj-ai-rag-llm-table-4-discursive-creditcard',
        'prj-ai-rag-llm-table-5-join-wo-transactions-2',
        'prj-ai-rag-llm-table-6-chunks',
        'prj-ai-rag-llm-table-7-cohere',
        'prj-ai-rag-llm-table-8-storytelling',
        'prj-ai-rag-llm-table-9-questions',
    )[8]
    vectorstore = get_vectorstore(
        name=vectorstore_name,
        embedding_function=embedding_function,
        database_client=database_client,
        create=False,
        dimension_count=1536
    )

    quiz_chain = CardTransactions.read_last_quiz()
    for prompt, answer in quiz_chain[-44:]:
        log(f'\033[96mPrompt: {prompt}\033[0m', end='\n')
        if '\n' in answer:
            answer = '\n\t' + answer.replace('\n', '\n\t')
        log(f'\033[96mGround truth: \033[0m{answer}', end='\n')
        log(f'\033[96mSimilar documents:\033[0m\n\t' +
            '\n\t'.join([doc.page_content for doc in query_documents(vectorstore, prompt, k=6)]))


    # Ask user for prompt until user types "exit"
    while True:
        # Example: "O cliente Carlos Eduardo Rodrigues (CPF: 80763159212) efetuou quais transaçoes em março de 2023?"
        prompt = input("\033[96mEnter a prompt (type 'exit' to quit):\033[0m ")
        if prompt == "exit":
            break

        # Search for similar documents
        result_documents = query_documents(vectorstore, prompt, k=6)
        for document in result_documents:
            # pprint(document.metadata)
            print(document.page_content, end='\n\n')
