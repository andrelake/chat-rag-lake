from time import perf_counter, sleep

from data_handler import load_document, chunk_data, insert_or_fetch_embeddings, get_pinecone_client, \
    get_embeddings_client, log
from env import PINECONE_INDEX_NAME, PINECONE_API_KEY, OPENAI_API_KEY
from openai_interaction_agent import asking_and_getting_answers

filepath = "data/texto.pdf"


def main() -> None:
    data = load_document(filepath)
    chunks = chunk_data(data)
    pinecone = get_pinecone_client(PINECONE_API_KEY)
    embeddings = get_embeddings_client(OPENAI_API_KEY)
    vector_store = insert_or_fetch_embeddings(PINECONE_INDEX_NAME, chunks, pinecone, embeddings)
    # chat_history_conversational = []

    i = 1
    print("Digite Sair ou Fim para encerrar.")
    while True:
        question = input(f"\nPergunta #{i}: ")
        if question.lower() in ["sair", "fim"]:
            print("Tchau!")
            sleep(2)
            break

        output = asking_and_getting_answers(vector_store, question, None)
        log(f"{output}")
        i += 1


if __name__ == "__main__":
    print("\nStarting...")
    log.verbose = True
    time_start = perf_counter()
    main()
    # delete_pinecone_index(index_name=PINECONE_INDEX_NAME)
    time_end = perf_counter()
    print(f"\nTime taken: {time_end - time_start:.2f} seconds")
