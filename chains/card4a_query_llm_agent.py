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


def get_retriever(vs):
    return vs.as_retriever(search_type="similarity", search_kwargs={"k": 15})


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(openai_api_key=OPENAI_API_KEY,
                      model_name=OPENAI_MODEL_NAME,
                      temperature=0,
                      streaming=True,
                      callbacks=[StreamingStdOutCallbackHandler()])


def get_prompt_template():
    return ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a helpful assistant. "
             "Use the tool \"transactions_search_tool\" as context to answer questions related to transactions. "
             "If the retrieved data has nothing to do with the question, just say \"Eu não tenho dados suficientes para responder essa instrução.\". "
             "Do not try to guess or make up an answer."
             "Question: {input}"),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )


def build_rag_chain():
    encoding = get_encoding(OPENAI_API_KEY)
    vs = get_vectorstore(PINECONE_INDEX_NAME, encoding)
    retriever = get_retriever(vs)
    llm = get_llm()
    prompt = get_prompt_template()
    retriever_tool = create_retriever_tool(
        retriever,
        "transactions_search_tool",
        "Use this tool to search in the context of transactions. "
        "Here you can find information about transactions. There are daily, monthly, and yearly summaries for each consumer and daily, monthly, and yearly summaries for the overall portfolio.",
    )
    tools = [retriever_tool]
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    return retriever, agent_executor


def ask_rag_chain(question, agent_executor):
    return agent_executor.invoke({"input": question})
