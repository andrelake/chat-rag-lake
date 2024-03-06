from time import perf_counter, sleep

from card4a_query_llm_lcel import ask_rag_chain


def chat_handler():
    i = 1
    print("Digite Sair ou Fim para encerrar.")
    while True:
        question = input(f"\nPergunta #{i}: ")
        if question.lower() in ["sair", "fim"]:
            print("Tchau!")
            sleep(2)
            break

        time_start = perf_counter()
        ask_rag_chain(question)
        time_end = perf_counter()
        print(f"\nTime taken: {time_end - time_start:.2f} seconds")
        i += 1


def main():
    chat_handler()


if __name__ == '__main__':
    main()
