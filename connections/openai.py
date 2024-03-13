from typing import List

from utils.utils import log

import tiktoken
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.globals import set_verbose, set_debug

# Data summarization
# from langchain.chains.summarize import load_summarize_chain


set_debug(log.verbose)
set_verbose(log.verbose)


models_pricing = {
    'text-embedding-3-small': 0.00002,
    'text-embedding-3-large': 0.00013,
    'text-embedding-ada-002': 0.00010,
}  # in dollars per token, as of 2024-02-26


def get_embeddings_client(api_key: str, model_name: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(openai_api_key=api_key, model=model_name)


def get_embedding_cost(documents: List[Document], model_name: str) -> None:
    encoding = tiktoken.encoding_for_model(model_name)
    total_tokens = sum([len(encoding.encode(document.page_content)) for document in documents])
    total_documents = len(documents)
    return {
        'model_name': model_name,
        'total_documents': total_documents,
        'total_tokens': total_tokens,
        'tokens_per_document': total_tokens / total_documents,
        'embedding_cost': total_tokens * models_pricing[model_name] / 1000
    }

def get_self_query_retriever(vectorstore, documents_description: str, documents_metadata: List[AttributeInfo]) -> SelfQueryRetriever:
    llm = ChatOpenAI(temperature=0)
    retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents=documents_description,
        metadata_field_info=documents_metadata,
        verbose=log.verbose
    )
    return retriever