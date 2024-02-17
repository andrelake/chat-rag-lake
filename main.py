from time import perf_counter

from data_handler import load_document, chunk_data, insert_or_fetch_embeddings

from env import PINECONE_INDEX_NAME
# from openai_interaction import get_interaction

filepath = "data/constituicao_federal.pdf"


def main() -> None:
    data = load_document(filepath)
    chunks = chunk_data(data)
    insert_or_fetch_embeddings(PINECONE_INDEX_NAME, chunks)
    # get_interaction(OPENAI_API_KEY)


if __name__ == "__main__":
    time_start = perf_counter()
    main()
    time_end = perf_counter()
    print(f"\nTime taken: {time_end - time_start:.2f} seconds")
