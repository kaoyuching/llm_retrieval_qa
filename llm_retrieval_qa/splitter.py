import os
from typing import List, Tuple, Optional

from langchain_core.documents import Document
from langchain_text_splitters import HTMLSectionSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_html(
    html_data: str,
    encoding: str = 'utf-8',
    sections_to_split: Optional[List[Tuple[str, str]]] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: List[str] = ["\n\n", "\n", ",", "."],
    ) -> List[Document]:
    r"""
        - filename: html filename
        - sections_to_split: html tags for splitting
            example = [
                ("h1", "Header 1"),
                ("h2", "Header 2"),
                ("h3", "Header 3"),
                ("h4", "Header 4"),
                ("h5", "Header 5"),
                ("table", 'table'),
            ]
        - chunk_size: text spliter's chunk size
        - chunk_overlap: number of texts to overlap
        - separators: separators used for splitting

    """
    html_splitter = HTMLSectionSplitter(sections_to_split)
    html_header_splits = html_splitter.split_text(html_data)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )
    split_docs = text_splitter.split_documents(html_header_splits)
    return split_docs
