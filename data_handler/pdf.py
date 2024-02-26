import re
from typing import List

from utils import log

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


def load_documents(filepath: str) -> List:
    log(f'Loading {filepath}')
    loader = PyPDFLoader(filepath)
    data = loader.load()
    for page in data:
        page.page_content = re.sub(r' +', ' ', page.page_content)
        page.page_content = re.sub(r'(?: \n)+(?<! )', ' \n', page.page_content)
    log(f'Data Loaded Successfully. Total pages: {len(data)}')
    return data


def redistribute_documents(data: List, chunk_size: int = 1600, chunk_overlap: int = 0) -> List:
    log(f'Starting chunk context')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(data)
    for i, chunk in enumerate(chunks):
        log(f'#{i+1} {chunk.page_content}')
    log(f'Total chunks: {len(chunks)}')
    return chunks
