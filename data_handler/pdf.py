import re
from typing import Optional, Any, Callable, List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


class Logger:
    def __init__(self, end: str = '\n', verbose: bool = False, **kwargs):
        self.end = end
        self.verbose = verbose
        self.kwargs = kwargs

    def __call__(self, message: str, **kwargs):
        kwargs = {'end': self.end, **self.kwargs, **kwargs}
        if self.verbose:
            print(message, **kwargs)


log = Logger()


def load_documents(filepath: str) -> List:
    log(f'Loading {filepath}')
    loader = PyPDFLoader(filepath)
    data = loader.load()
    for page in data:
        page.page_content = re.sub(r' +', ' ', page.page_content)
        page.page_content = re.sub(r'(?: \n)+(?<! )', ' \n', page.page_content)
    log(f'Data Loaded Successfully. Total pages: {len(data)}')
    return data


def redistribute_chunks(data: List, chunk_size: int = 1600, chunk_overlap: int = 0) -> List:
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
