from langchain_community.vectorstores import Pinecone as PineconeVS
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from env import OPENAI_MODEL_NAME, OPENAI_API_KEY, PINECONE_INDEX_NAME


def get_encoding(api_key: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(openai_api_key=api_key,
                            model="text-embedding-3-small")


def get_vectorstore(index_name, encoding):
    return PineconeVS.from_existing_index(index_name, encoding)


def get_retriever(vs):
    return vs.as_retriever(search_type="similarity", search_kwargs={"k": 5})


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(openai_api_key=OPENAI_API_KEY,
                      model_name=OPENAI_MODEL_NAME,
                      temperature=0,
                      streaming=True,
                      callbacks=[StreamingStdOutCallbackHandler()])


def get_prompt_template():
    template = """
        Você é um assistente muito bom e está ajudando um gerente de contas de um banco. 
        Responda as questões baseado no seguinte contexto: 
            {context} 
        Se não tiver informação suficiente para responder a questão, responda: 
        "Nao consegui informação suficiente para responder" 
        Questão: {input}
    """
    return ChatPromptTemplate.from_template(template)


def build_rag_chain():
    encoding = get_encoding(OPENAI_API_KEY)
    vs = get_vectorstore(PINECONE_INDEX_NAME, encoding)
    retriever = get_retriever(vs)
    llm = get_llm()
    prompt = get_prompt_template()
    chain = (
            {"context": retriever, "input": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return retriever, chain


def ask_rag_chain(question, rag_chain):
    return rag_chain.invoke(question)


# if __name__ == '__main__':
#     encoding = get_encoding(OPENAI_API_KEY)
#     vs = get_vectorstore(PINECONE_INDEX_NAME, encoding)
#     retriever = get_retriever(vs)
#     resp = retriever.invoke("Total gasto por Levi Fogaça em abril de 2023")
#     pprint(resp)
