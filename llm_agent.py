from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import Pinecone as PineconeVS
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from env import OPENAI_MODEL_NAME, OPENAI_API_KEY, PINECONE_INDEX_NAME


def get_encoding(api_key: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(openai_api_key=api_key,
                            model="text-embedding-3-small")


def get_vectorstore(index_name, encoding):
    return PineconeVS.from_existing_index(index_name, encoding)


def get_retriever(vs, k, filter):
    return vs.as_retriever(search_type="similarity", search_kwargs={"k": k, 'filter': filter})


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(openai_api_key=OPENAI_API_KEY,
                      model_name=OPENAI_MODEL_NAME,
                      temperature=0,
                      streaming=True,
                      callbacks=[StreamingStdOutCallbackHandler()])


# Tools
## transactions_search_tool
## consumer_daily_transactions_search_tool
## consumer_monthly_transactions_search_tool
## consumer_yearly_transactions_search_tool
## portfolio_daily_transactions_search_tool
## portfolio_monthly_transactions_search_tool
## portfolio_yearly_transactions_search_tool

def get_prompt_template():
    return ChatPromptTemplate.from_messages(
        [
            ('system',
             'You are a helpful assistant. '
             'Use the tools "transactions_search_tool", "consumer_daily_transactions_search_tool", "consumer_monthly_transactions_search_tool", "consumer_yearly_transactions_search_tool", "portfolio_daily_transactions_search_tool", "portfolio_monthly_transactions_search_tool", and "portfolio_yearly_transactions_search_tool" '
             'wisely to search in the context of transactions considering which data aggregations are needed to answer the question. '
             'If you need to call multiple tools, do it without calling them in parallel. '
             'If the retrieved data is not related to the question, just say "Eu não tenho dados suficientes para responder essa instrução.". '
             'Do not try to guess and absolutely never make up data. '
             'In our language, portfolio means "carteira". '
             'Question: {input}'),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )


def build_rag_chain():
    encoding = get_encoding(OPENAI_API_KEY)
    vs = get_vectorstore(PINECONE_INDEX_NAME, encoding)
    llm = get_llm()
    prompt = get_prompt_template()
    tools = [
        create_retriever_tool(
            get_retriever(vs, 6, {'data_granularity': 'transaction_id'}),
            'transactions_search_tool',
            'Use this tool to search in the context of transactions if details of individual transactions are needed such as a specific transaction\'s ID, amount, date, and location of a transaction. '
            'Query example: <query>O cliente Lucas Viana (CPF: 92106574894) efetuou a transação de ID "0" com cartão de crédito PLATINUM de R$ 1536.10 no dia 2 do mês de janeiro do ano de 2023 (02/01/2023) para o estabelecimento "Caldeira"</query>'
        ),
        create_retriever_tool(
            get_retriever(vs, 6, {'data_granularity': 'transaction_year/transaction_month/transaction_day/portfolio_id/consumer_id'}),
            'consumer_daily_transactions_search_tool',
            'Use this tool to search in the context of transactions if details of daily transactions for a specific consumer are needed such as the number of transactions, total amount of transactions, average transaction amount, smallest and largest transaction amounts. '
            'Query example: <query>Resumo diário das transações do cliente Lucas Viana (CPF: 92106574894) no dia 2 do mês de janeiro do ano de 2023 (02/01/2023), com uma contagem total de 2 transações e valor total de R$ 6023.44, o valor médio é de R$ 3011.72, o valor da maior transação é de R$ 4487.33 e o valor da menor transação é de R$ 1536.10. Dentre elas foram realizadas 1 com cartão PLATINUM, 1 com cartão INTERNACIONAL.</query>'
        ),
        create_retriever_tool(
            get_retriever(vs, 6, {'data_granularity': 'transaction_year/transaction_month/portfolio_id/consumer_id'}),
            'consumer_monthly_transactions_search_tool',
            'Use this tool to search in the context of transactions if details of monthly transactions for a specific consumer are needed such as the number of transactions, total amount of transactions, average transaction amount, smallest and largest transaction amounts.'
            'Query example: <query>Resumo mensal das transações do cliente Lucas Viana (CPF: 92106574894) no mês de janeiro do ano de 2023, com uma contagem...</query>',
        ),
        create_retriever_tool(
            get_retriever(vs, 6, {'data_granularity': 'transaction_year/portfolio_id/consumer_id'}),
            'consumer_yearly_transactions_search_tool',
            'Use this tool to search in the context of transactions if details of yearly transactions for a specific consumer are needed such as the number of transactions, total amount of transactions, average transaction amount, smallest and largest transaction amounts.'
            'Query example: <query>Sumário anual de transações do cliente Lucas Viana (CPF: 92106574894) no ano de 2023, com uma contagem...</query>',
        ),
        create_retriever_tool(
            get_retriever(vs, 6, {'data_granularity': 'transaction_year/transaction_month/transaction_day/portfolio_id'}),
            'portfolio_daily_transactions_search_tool',
            'Use this tool to search in the context of transactions if details of daily transactions for the entire portfolio are needed such as the number of transactions, total amount of transactions, average transaction amount, smallest and largest transaction amounts.'
            'Query example: <query>Sumário diário de transações de todos os clientes da carteira "0" no dia 2 do mês de janeiro do ano de 2023 (02/01/2023), com uma contagem...</query>'
        ),
        create_retriever_tool(
            get_retriever(vs, 6, {'data_granularity': 'transaction_year/transaction_month/portfolio_id'}),
            'portfolio_monthly_transactions_search_tool',
            'Use this tool to search in the context of transactions if details of monthly transactions for the entire portfolio are needed such as the number of transactions, total amount of transactions, average transaction amount, smallest and largest transaction amounts.'
            'Query example: <query>Sumário mensal de transações de todos os clientes da carteira "0" para o mês de janeiro do ano de 2023 (01/2023), com uma contagem...<query>'
        ),
        create_retriever_tool(
            get_retriever(vs, 6, {'data_granularity': 'transaction_year/portfolio_id'}),
            'portfolio_yearly_transactions_search_tool',
            'Use this tool to search in the context of transactions if details of yearly transactions for the entire portfolio are needed such as the number of transactions, total amount of transactions, average transaction amount, smallest and largest transaction amounts.'
            'Query example: <query>Sumário anual de transações de todos os clientes da carteira "0" para o ano de 2023, com uma contagem...</query>'
        ),
    ]
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, max_iterations=15, verbose=True)

    return agent_executor


def ask_rag_chain(question, agent_executor):
    return agent_executor.invoke({"input": question})
