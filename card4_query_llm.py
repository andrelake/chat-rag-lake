from langchain_community.vectorstores import Pinecone as PineconeVS
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from env import OPENAI_MODEL_NAME, OPENAI_API_KEY, PINECONE_INDEX_NAME


def get_encoding(api_key: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(openai_api_key=api_key)


def get_vectorstore(index_name, encoding):
    print(f'Getting info from index: {index_name}')
    return PineconeVS.from_existing_index(index_name, encoding)


def get_retriever(vs):
    return vs.as_retriever(search_type="similarity", search_kwargs={"k": 5})


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(openai_api_key=OPENAI_API_KEY,
                      model_name=OPENAI_MODEL_NAME,
                      temperature=0.5,
                      streaming=True,
                      callbacks=[StreamingStdOutCallbackHandler()])


def get_prompt() -> ChatPromptTemplate:
    system_template = (
        """
        You are a helpful bank assistant that helps a account manager to answer questions.
        You can only use Portuguese, from Brazil, to read and answer.
        You should only using the agent tool if asked about something related to transactions.
        You also can use the following context and chat history to answer the questions. 
        Combine chat history and the last user's question to create a new independent question. 
        Context: {context}
        Chat history: {chat_history} 
        Question: {input}
        """
    )

    return ChatPromptTemplate.from_messages(
        [SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'chat_history', 'input'],
                                                           template=system_template)),
         MessagesPlaceholder(variable_name='chat_history', optional=True),
         HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}')),
         MessagesPlaceholder(variable_name='agent_scratchpad')]
    )


def build_rag_chain():
    encoding = get_encoding(OPENAI_API_KEY)
    vs = get_vectorstore(PINECONE_INDEX_NAME, encoding)
    prompt = get_prompt()
    llm = get_llm()
    return (
            {"context": get_retriever(vs), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )


def ask_rag_chain(question, rag_chain=build_rag_chain()):
    return rag_chain.invoke(question)
