from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Pinecone as PineconeVS
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
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


def get_prompt_from_hub(prompt_template_name: str) -> ChatPromptTemplate:
    return hub.pull(prompt_template_name)


def build_rag_chain():
    combine_docs_chain = create_stuff_documents_chain(
        get_llm(), get_prompt_from_hub("langchain-ai/retrieval-qa-chat")
    )
    encoding = get_encoding(OPENAI_API_KEY)
    vs = get_vectorstore(PINECONE_INDEX_NAME, encoding)
    retriever = get_retriever(vs)

    return create_retrieval_chain(retriever, combine_docs_chain)


def ask_rag_chain(question, rag_chain=build_rag_chain()):
    return rag_chain.invoke(question)
