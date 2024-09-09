import os
import io
from io import IOBase
from typing import List, Tuple, Optional, Union
import json
import copy
import numpy as np

from pdfminer.high_level import extract_text
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTText

from langchain_core.documents import Document
from langchain_text_splitters import HTMLSectionSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter, RecursiveJsonSplitter

from llm_retrieval_qa.data_parser.html_parser import DataFromUrl


r"""
Advanced RAG: https://hackmd.io/@YungHuiHsu/rkqGpCDca
"""


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


class DataSplitter():
    def __init__(
        self,
        file_src: Union[str, IOBase],
        file_ext: str,
        encoding: str = 'utf-8',
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: List[str] = ["\n\n", "\n", ",", "."],
        keyword_extract: bool = False,
    ):
        self.file_src = file_src
        # get file ext
        self.ext = file_ext
        self.encoding = encoding
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        )

        self.kw_model = None
        if keyword_extract:
            from keybert import KeyBERT
            self.kw_model = KeyBERT()

    def load_data(self, file_src: Union[str, IOBase]) -> List[Document]:
        if self.ext == ".html":
            data = self._load_html(file_src, encoding=self.encoding)
        elif self.ext == ".pdf":
            data = self._load_pdf(file_src)
        elif self.ext == ".txt":
            data = self._load_txt(file_src, encoding=self.encoding)
        elif self.ext in [".doc", ".docx"]:
            data = self._load_doc(file_src)
        elif self.ext == ".json":
            data = self._load_json(file_src)
        else:
            raise ValueError("Invalid file format.")

        if self.kw_model:
            data = self.keyword_extraction(data, top_n=10)
        return data

    def split(self, data: List[Document]):
        split_docs = self.text_splitter.split_documents(data)
        return split_docs

    def _load_html(self, file_src: Union[str, IOBase], encoding: str = 'utf-8') -> List[Document]:
        r"""
        file_src: local file path or BytesIO
        """
        if isinstance(file_src, IOBase):
            data = file_src.read()
            data = data.decode(encoding=encoding)
        else:
            with open(file_src, 'r', encoding=encoding) as f:
                data = f.read()

        sections_to_split = [
            ("h1", "Header 1"),
            ("h2", "Header 2"),
            ("h3", "Header 3"),
            ("h4", "Header 4"),
            ("h5", "Header 5"),
            ("table", 'table'),
        ]

        html_splitter = HTMLSectionSplitter(sections_to_split)
        data = html_splitter.split_text(data)
        return data

    def _load_pdf(self, file_src: Union[str, IOBase]) -> List[Document]:
        pdf_pages = extract_pages(file_src)

        special_text = ["\xa0", "\t"]
        data = []
        for i, page_layout in enumerate(pdf_pages):
            page_num = i + 1
            elements = []
            for element in page_layout:
                if isinstance(element, LTText):
                    elements.append(element.get_text())
            text = ''.join(elements)
            for special in special_text:
                text = text.replace(special, " ")
            doc = Document(page_content=text, metadata={"page": page_num})
            data.append(doc)
        return data

    def _load_txt(self, file_src: str, encoding: str = 'utf-8'):
        if isinstance(file_src, IOBase):
            data = file_src.read()
            data = data.decode(encoding=encoding)
        else:
            with open(file_src, 'r', encoding=encoding) as f:
                data = f.read()
        data = [Document(page_content=data)]
        return data

    def _load_doc(self, file_src: Union[str, IOBase], encoding: str = 'utf-8'):
        return

    def _load_json(self, file_src: Union[str, IOBase], encoding: str = 'utf-8'):
        if isinstance(file_src, IOBase):
            data = file_src.read()
            data = json.loads(data.decode(encoding=encoding))
        else:
            with open(file_src, 'r', encoding=encoding) as f:
                data = json.load(f)
        
        if isinstance(data, list):
            docs = [Document(page_content=json.dumps(x)) for x in data]
        else:
            docs = [Document(page_content=json.dumps(data))]
        return docs

    def keyword_extraction(self, docs: List[Document], top_n: int = 10) -> List[Document]:
        new_docs = copy.deepcopy(docs)
        for i, _doc in enumerate(docs):
            doc = _doc.page_content
            keywords = self.kw_model.extract_keywords(doc, top_n=top_n)
            keywords = [x[0] for x in keywords]
            new_docs[i].metadata = {**new_docs[i].metadata, 'keywords': keywords}

        # unique, counts = np.unique(res, return_counts=True)
        # summary = list(zip(unique, counts))
        # summary = sorted(summary, key=lambda x: x[1], reverse=True)
        return new_docs

    def __call__(self):
        data = self.load_data(self.file_src)
        split_docs = self.split(data)
        return split_docs
