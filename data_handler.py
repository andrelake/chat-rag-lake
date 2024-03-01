import os
from shutil import rmtree
from typing import Optional, Any, Callable, List, Dict
import random
import json

from utils import Logger, log

import pandas as pd
import numpy as np
from faker import Faker
from fastavro import block_reader, writer
from langchain_core.documents import Document
from langchain.chains.query_constructor.base import AttributeInfo


def generate_documents(
    df: pd.DataFrame,
    where: Optional[Callable[[Any], bool]] = None,
    group_by: Optional[List[str]] = None,
    order_by: Optional[List[str]] = None,
    limit: Optional[int] = None,
    parse_content_header: Optional[Callable[[Any], str]] = None,
    parse_content_body: Optional[Callable[[Any], str]] = None,
    parse_metadata: Optional[Callable[[Any], dict]] = lambda record: dict(record),
    log: Optional[Logger] = log
) -> List[Document]:
    log('Generating documents...')
    if where:
        df = df[where(df)].copy()
    if group_by:
        if parse_content_body:
            df['_page_content_body'] = df.apply(lambda record: Document(page_content=parse_content_body(record)), axis=1)
            df = df.groupby(group_by).agg({'_page_content_body': '\n'.join}).reset_index()
        else:
            df['_page_content_body'] = ''
            df = df.groupby(group_by).agg({'_page_content_body': ''.join}).reset_index()
    else:
        df['_page_content_body'] = ''
    df['_page_content_header'] = df.apply(lambda record: parse_content_header(record), axis=1)
    df['_metadata'] = df.apply(lambda record: parse_metadata(record), axis=1)
    if order_by:
        order_by = (group_by or []) + [col for col in order_by if col not in group_by]
        df = df.sort_values(order_by)
    df = df.drop(columns=[col for col in df.columns if col not in ['_page_content_header', '_page_content_body', '_metadata']])
    if limit:
        df = df.head(limit)
    documents = df.apply(
        lambda record: Document(
            page_content=record['_page_content_header'] + record['_page_content_body'],
            metadata=record['_metadata']
        ),
        axis=1
    ).to_list()
    return documents


def write_avro(df: pd.DataFrame, path: str, schema: Dict, partitionBy: Optional[List[str]], log: Optional[Logger] = log):
    # Delete old data
    log(f'Deleting old data from `{path}`...')
    rmtree(path, ignore_errors=True)

    # Save data to disk as partitioned Avro files
    log(f'Writing Avro files to `{path}`...')
    if partitionBy:
        for group, group_df in df.groupby(partitionBy):
            file_path = os.path.join(path, *[f'{k}={v}' for k, v in zip(partitionBy, group)], 'data.avro')
            with open(file_path, 'wb') as fo:
                writer(fo, schema, group_df.to_dict(orient='records'))
    else:
        file_path = os.path.join(path, 'data.avro')
        with open(file_path, 'wb') as fo:
            writer(fo, schema, df.to_dict(orient='records'))


def write_orc(df: pd.DataFrame, path: str, partitionBy: Optional[List[str]], compression: Optional[str] = 'zstd', log: Optional[Logger] = log):
    # Delete old data
    log(f'Deleting old data from `{path}`...')
    rmtree(path, ignore_errors=True)

    # Convert pandas DataTypes to ORC supported DataTypes
    dtype_translation = {
        'uint8': np.int16,
        'uint16': np.int32,
        'uint32': np.int64,
        'uint64': np.int64,
        'category': pd.StringDtype(),
    }
    df = df.copy()
    for col in df.columns:
        dtype = str(df[col].dtype)
        if dtype in dtype_translation:
            df[col] = df[col].astype(dtype_translation[dtype])


    # Save data to disk as partitioned ORC files
    log(f'Writing ORC files to `{path}`...')
    if partitionBy:
        for group, group_df in df.groupby(partitionBy):
            file_path = os.path.join(path, *[f'{k}={v}' for k, v in zip(partitionBy, group)], f'data.{compression}.orc')
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            group_df.to_orc(file_path, index=False, engine_kwargs={'compression': 'zstd'})
    else:
        file_path = os.path.join(path, f'data.{compression}.orc')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_orc(file_path, index=False, engine_kwargs={'compression': 'zstd'})


def read_orc(path: str, log: Optional[Logger] = log) -> pd.DataFrame:
    log(f'Reading ORC files from `{path}`...')
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.orc'):
                file_path = os.path.join(root, file)
                df = pd.read_orc(file_path)
                return df


def write_json(df: pd.DataFrame, path: str, log: Optional[Logger] = log):
    # Save data to disk as JSONL file
    log(f'Writing JSONL file to `{path}`...')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_json(path, orient='records')


def read_json(path: str, log: Optional[Logger] = log) -> pd.DataFrame:
    log(f'Reading JSONL file from `{path}`...')
    df = pd.read_json(path)
    return df
