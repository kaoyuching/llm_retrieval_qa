import sys
sys.path.append("./")
import os
import numpy as np
import pandas as pd

from langchain_core.documents import Document
from langchain_text_splitters import HTMLSectionSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

import torch
from transformers import AutoTokenizer, TextStreamer, pipeline, GPTQConfig

# version: write with transformers generation
# https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM
from transformers import AutoModelForCausalLM

from splitter import split_html
from llm_prompt import PromptLlama2, PromptLlama3


r'''
llama2: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-2
llama2 prompt template: https://github.com/meta-llama/llama/blob/main/llama/generation.py#L284-L395
llama3: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/
'''


filename = './example_files/nvidia_doc.html'
# filename = './example_files/sql_alchemy_doc_all.html'
# filename = './example_files/germany_beginner.html'
with open(filename, 'r', encoding='utf-8') as f:
    html_doc = f.read()

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
    ("h4", "Header 4"),
    ("h5", "Header 5"),
    ("table", 'table'),
]

html_splitter = HTMLSectionSplitter(headers_to_split_on)

html_header_splits = html_splitter.split_text(html_doc)

chunk_size = 1000 # 800
chunk_overlap = 50 # 30
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", ',', '.'],
)

# Split
splits = text_splitter.split_documents(html_header_splits)


# model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
model_name = "GanymedeNil/text2vec-large-chinese"
hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    # model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    encode_kwargs={'normalize_embeddings': True},
)


# vector store
faiss_db = FAISS.from_documents(
    splits,
    hf_embeddings,
#     normalize_L2=True,
    distance_strategy=DistanceStrategy.COSINE,
)

retriever = faiss_db.as_retriever(
    search_type='similarity',  # mmr
    search_kwarg={'k': 20},
)


# llm model
# llm_model_name = "yentinglin/Taiwan-LLM-7B-v2.1-chat"
# llm_model_name = "../models/llama2-7b-chat"
llm_model_name = "../models/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

model = AutoModelForCausalLM.from_pretrained(
    llm_model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="cuda",
)

hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    num_beams=1,
    max_new_tokens=2048,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    # temperature=0.0,  # 0.7 #TODO: temperature < 0.1
    do_sample=False,  # False if temperature is 0
    # top_p=0.9,
    repetition_penalty=1.1,
    return_full_text=False,
    # truncation=True,
#     streamer=streamer,
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)


# prompt
# llama_prompt = PromptLlama2()
llama_prompt = PromptLlama3()

# llama3
# system = """Use the following context to answer the user's question.
# If you don't know the answer or the question is not directly related to the conext, you should answer you don't know and don't generate any answers."""

# instruction = """Please answer the {question} directly according to the context: {context}"""


# llama2
system = """You serve as a assistant specialized in answering questions according to the given context.
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up any answers."""

instruction = """context: {context}
question: {question}"""



llama_prompt.set_system_prompt(system_prompt=system)
prompt_template_fn, full_prompt = llama_prompt.get_template(instruction)
print(full_prompt)

from utils import QAChain

qa_chain = QAChain(llm, faiss_db, prompt_template_fn, top_k=10, return_source_documents=True)


print("=================start=====================")
res = qa_chain("How to say 'thank you' in germany?")
print(f"Question: {res['query']}\nAnswer: {res['result']}")
print("---------------------")
print(f"docs: {res['source_documents']}")
print("=================")

res = qa_chain("Can you introduce some german food?")
print(f"Question: {res['query']}\nAnswer: {res['result']}")
print("---------------------")
print(f"docs: {res['source_documents']}")
print("=================")

res = qa_chain("How to use and_ in SQLAlchemy?")
print(f"Question: {res['query']}\nAnswer: {res['result']}")
print("---------------------")
print(f"docs: {res['source_documents']}")
print("=================")

res = qa_chain("How to use colume with SQLAlchemy?")
print(f"Question: {res['query']}\nAnswer: {res['result']}")
print("---------------------")
print(f"docs: {res['source_documents']}")
print("=================")

res = qa_chain("What is the compute capability?")
print(f"Question: {res['query']}\nAnswer: {res['result']}")
print("---------------------")
print(f"docs: {res['source_documents']}")
print("=================")

res = qa_chain("What is TensorRT?")
print(f"Question: {res['query']}\nAnswer: {res['result']}")
print("---------------------")
print(f"docs: {res['source_documents']}")
print("=================")

res = qa_chain("Can you explain what is TensorRT?")
print(f"Question: {res['query']}\nAnswer: {res['result']}")
print("---------------------")
print(f"docs: {res['source_documents']}")
print("=================")

res = qa_chain("How to bulid TensorRT engine?")
print(f"Question: {res['query']}\nAnswer: {res['result']}")
print("---------------------")
print(f"docs: {res['source_documents']}")
print("=================")

res = qa_chain("How to bulid TensorRT engine with python?")
print(f"Question: {res['query']}\nAnswer: {res['result']}")
print("---------------------")
print(f"docs: {res['source_documents']}")
print("=================")

res = qa_chain("How to bulid TensorRT engine with python from onnx model?")
print(f"Question: {res['query']}\nAnswer: {res['result']}")
print("---------------------")
print(f"docs: {res['source_documents']}")
