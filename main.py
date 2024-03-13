from time import perf_counter, sleep

from llm_agent import ask_rag_chain, build_rag_chain


def chat_handler(chain):
    i = 1
    print("Digite Sair ou Fim para encerrar.")
    while True:
        question = input(f"\nPergunta #{i}: ")
        if question.lower() in ["sair", "fim"]:
            print("Tchau!")
            sleep(2)
            break

        time_start = perf_counter()
        resp = ask_rag_chain(question, chain)
        print(f"\nResposta #{i}: ")
        print(resp["output"])
        time_end = perf_counter()
        print(f"\nTime taken: {time_end - time_start:.2f} seconds")
        i += 1


def main():
    # retriever, rag_chain_with_source = build_rag_chain()
    rag_chain_with_source = build_rag_chain()
    chat_handler(rag_chain_with_source)


if __name__ == '__main__':
    main()
