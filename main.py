from time import perf_counter, sleep

from data_handler import load_document, chunk_data, insert_or_fetch_embeddings
from env import PINECONE_INDEX_NAME
from openai_interaction import asking_and_getting_answers

filepath = "data/texto.pdf"


def main() -> None:
    data = load_document(filepath)
    chunks = chunk_data(data)
    vector_store = insert_or_fetch_embeddings(PINECONE_INDEX_NAME, chunks)
    chat_history = []

    i = 1
    print("Digite Sair ou Fim para encerrar.")
    while True:
        question = input(f"\nPergunta #{i}: ")
        if question.lower() in ["sair", "fim"]:
            print("Tchau!")
            sleep(2)
            break

        result, chat_history = asking_and_getting_answers(vector_store, question, chat_history)
        print(f"\nAnswer : {result}")
        print(f"\nChat History : {chat_history}")
        print(f"\n {'-' * 50}")
        i += 1


if __name__ == "__main__":
    print("\nStarting...")
    time_start = perf_counter()
    main()
    # delete_pinecone_index(index_name=PINECONE_INDEX_NAME)
    time_end = perf_counter()
    print(f"\nTime taken: {time_end - time_start:.2f} seconds")