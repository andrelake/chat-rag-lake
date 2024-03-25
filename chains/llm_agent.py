from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import Pinecone as PineconeVS
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from env import OPENAI_MODEL_NAME, OPENAI_API_KEY, PINECONE_INDEX_NAME, OPENAI_EMBEDDING_MODEL_NAME


def get_encoding(api_key: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(openai_api_key=api_key,
                            model=OPENAI_EMBEDDING_MODEL_NAME,
                            dimensions=3072)


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
             'You must translate the question to English to build the query and must answer only in Portuguese, from Brazil. '
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
            'Query example: <query>Customer Levi Fogaça (CPF: 45281639773) made a credit card (BLACK) transaction of R$ 3973.10 (ID \"8980\") on novembro 16, 2023 (16/11/2023) at the establishment \"Martins\"</query>'
        ),
        create_retriever_tool(
            get_retriever(vs, 6, {'data_granularity': 'transaction_year/transaction_month/transaction_day/portfolio_id/consumer_id'}),
            'consumer_daily_transactions_search_tool',
            'Use this tool to search in the context of transactions if details of daily transactions for a specific consumer are needed such as the number of transactions, total amount of transactions, average transaction amount, smallest and largest transaction amounts. '
            'Query example: <query>Daily summary of credit card transactions for customer Levi Fogaça (CPF: 45281639773) on novembro 16, 2023, (16/11/2023), with a total count of 1 transactions and a total value of R$ 3973.10, the average value is R$ 3973.10, the value of the largest transaction is R$ 3973.10 and the value of the smallest transaction is R$ 3973.10. Of these, 1 were made witha BLACK card.</query>'
        ),
        create_retriever_tool(
            get_retriever(vs, 6, {'data_granularity': 'transaction_year/transaction_month/portfolio_id/consumer_id'}),
            'consumer_monthly_transactions_search_tool',
            'Use this tool to search in the context of transactions if details of monthly transactions for a specific consumer are needed such as the number of transactions, total amount of transactions, average transaction amount, smallest and largest transaction amounts.'
            'Query example: <query>'
            'Monthly summary of credit card transactions for customer Levi Fogaça (CPF: 45281639773) for the entire month of novembro, 2023 (11/2023), with a total count of 21 transactions and a total value of R$ 41154.85, the average value is R$ 1959.75, the value of the largest transaction is R$ 3973.10 and the value of the smallest transaction is R$ 326.33. Of these, 4 were made with a BLACK card, 12 were made with a GOLD card, 1 were made with a PLATINUM card, 3 were made with a STANDARD card, 1 were made with an INTERNATIONAL card.'
            '</query>',
        ),
        create_retriever_tool(
            get_retriever(vs, 6, {'data_granularity': 'transaction_year/portfolio_id/consumer_id'}),
            'consumer_yearly_transactions_search_tool',
            'Use this tool to search in the context of transactions if details of yearly transactions for a specific consumer are needed such as the number of transactions, total amount of transactions, average transaction amount, smallest and largest transaction amounts.'
            'Query example: <query>'
            'Annual summary of credit card transactions for customer Levi Fogaça (CPF: 45281639773) for the entire year of 2023, with a total count of 297 transactions and a total value of R$ 769380.30, the average value is R$ 2590.51, the value of the largest transaction is R$ 4981.62 and the value of the smallest transaction is R$ 28.80. Of these, 57 were made with a BLACK card, 58 were made with a GOLD card, 53 were made with a PLATINUM card, 69 were made with a STANDARD card, 60 were made with an INTERNATIONAL card.'
            '</query>',
        ),
        create_retriever_tool(
            get_retriever(vs, 6, {'data_granularity': 'transaction_year/transaction_month/transaction_day/portfolio_id'}),
            'portfolio_daily_transactions_search_tool',
            'Use this tool to search in the context of transactions if details of daily transactions for the entire portfolio are needed such as the number of transactions, total amount of transactions, average transaction amount, smallest and largest transaction amounts.'
            'Query example: <query>'
            'Daily summary of credit card transactions for all customers of the portfolio "0" on novembro 16, 2023, (16/11/2023), with a total count of 11 transactions and a total value of R$ 21569.09, the average value is R$ 1960.83, the value of the largest transaction is R$ 4069.05 and the value of the smallest transaction is R$ 457.07. Of these, 5 were made with a BLACK card, 3 were made with a PLATINUM card, 2 were made with a STANDARD card, 1 were made with an INTERNATIONAL card.'
            '</query>'
        ),
        create_retriever_tool(
            get_retriever(vs, 6, {'data_granularity': 'transaction_year/transaction_month/portfolio_id'}),
            'portfolio_monthly_transactions_search_tool',
            'Use this tool to search in the context of transactions if details of monthly transactions for the entire portfolio are needed such as the number of transactions, total amount of transactions, average transaction amount, smallest and largest transaction amounts.'
            'Query example: <query>'
            'Monthly summary of credit card transactions for all customers of the portfolio "0" for the entire month of novembro, 2023 (11/2023), with a total count of 232 transactions and a total value of R$ 532135.10, the average value is R$ 2293.69, the value of the largest transaction is R$ 4940.37 and the value of the smallest transaction is R$ 5.94. Of these, 52 were made with a BLACK card, 47 were made with a GOLD card, 49 were made with a PLATINUM card, 47 were made with a STANDARD card, 37 were made with an INTERNATIONAL card.'
            '<query>'
        ),
        create_retriever_tool(
            get_retriever(vs, 6, {'data_granularity': 'transaction_year/portfolio_id'}),
            'portfolio_yearly_transactions_search_tool',
            'Use this tool to search in the context of transactions if details of yearly transactions for the entire portfolio are needed such as the number of transactions, total amount of transactions, average transaction amount, smallest and largest transaction amounts.'
            'Query example: <query>'
            'Annual summary of credit card transactions for all customers of the portfolio "0" for the entire year of 2023, with a total count of 3004 transactions and a total value of R$ 7558624.98, the average value is R$ 2516.19, the value of the largest transaction is R$ 4999.94 and the value of the smallest transaction is R$ 1.25. Of these, 635 were made with a BLACK card, 593 were made with a GOLD card, 589 were made with a PLATINUM card, 593 were made with a STANDARD card, 594 were made with an INTERNATIONAL card.'
            '</query>'
        ),
    ]
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, max_iterations=15, verbose=True)

    return agent_executor


def ask_rag_chain(question, agent_executor):
    return agent_executor.invoke({"input": question})
