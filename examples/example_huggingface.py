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
from transformers import AutoTokenizer, TextStreamer, pipeline
from transformers import GPTQConfig, BitsAndBytesConfig, QuantoConfig

# version: write with transformers generation
# https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM
from transformers import AutoModelForCausalLM

from llm_retrieval_qa.splitter import split_html
from llm_retrieval_qa.pipeline.llm_prompt import PromptLlama2, PromptLlama3, PromptPhi3, Message


r'''
llama2: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-2
llama2 prompt template: https://github.com/meta-llama/llama/blob/main/llama/generation.py#L284-L395
llama3: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/
'''


filename = './example_files/sql_alchemy_doc_all.html'
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


# model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model_name = "GanymedeNil/text2vec-large-chinese"
# model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    encode_kwargs={'normalize_embeddings': True},
)


# vector store
# TODO: vector store and similarity search
# import faiss

# embedding_list = hf_embeddings.embed_documents(splits)
# embedding_dim = len(embedding_list)
# index = faiss.IndexFlatL2(embedding_dim)
# index.add(embedding_list)


faiss_db = FAISS.from_documents(
    splits,
    hf_embeddings,
    # normalize_L2=True,
    distance_strategy=DistanceStrategy.COSINE,
)

# TODO: langchain retriever
retriever = faiss_db.as_retriever(
    # search_type='similarity',  # mmr
    search_type='mmr',  # mmr
    search_kwarg={'k': 20},
)


# llm model
# llm_model_name = "yentinglin/Taiwan-LLM-7B-v2.1-chat"
# llm_model_name = "../models/llama2-7b-chat"
# llm_model_name = "../models/llama2-7b-chat-8bit"
# llm_model_name = "../models/Llama-2-7b-chat-hf"
llm_model_name = "microsoft/Phi-3-mini-4k-instruct"
# llm_model_name = "../models/Meta-Llama-3-8B-Instruct"
# llm_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# llm_model_name = "../models/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
# tokenizer.save_pretrained("../models/llama2-7b-chat-8bit")

# TODO: quantization
quantization_config = BitsAndBytesConfig(
    # load_in_8bit=True,
    # llm_int8_enable_fp32_cpu_offload=False,
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
)

print("load model")
# from compression import load_compress_model
# model = load_compress_model(llm_model_name, "cuda", torch.float16, False, use_safetensors=True)

model = AutoModelForCausalLM.from_pretrained(
    llm_model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="cuda",
    # quantization_config=quantization_config,
    low_cpu_mem_usage=True,
    # use_safetensors=True,
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
    temperature=0.0,  # 0.7 #TODO: temperature < 0.1
    do_sample=False,  # False if temperature is 0
    # top_p=0.9,
    repetition_penalty=1.1,
    return_full_text=False,
    # low_memory=True,
    # truncation=True,
#     streamer=streamer,
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)


# llama3
# system = """Use the following context to answer the user's question.
# If you don't know the answer or the question is not directly related to the context, you should answer you don't know and don't generate any answers."""

# instruction = """Please answer the {question} directly according to the context: {context}"""


# llama2
# system = """You serve as a assistant specialized in answering questions with the given context.
# If the following context is not directly related to the question, you must say that you don't know.
# Don't try to make up any answers. No potential connection and no guessing."""

# instruction = """Base on the following context: {context}, please answer {question}.
# If the question is not directly related to the description, you should answer you don't know."""


# phi3
system = """You serve as a assistant specialized in answering questions with the given context.
If the following context is not directly related to the question, you must say that you don't know.
Don't try to make up any answers. No potential connection and no guessing."""

instruction = """Base on the following context: {context}, please answer {question}.
If the question is not directly related to the description, you should answer you don't know."""

llama_prompt = PromptPhi3(system)
messages = [Message("user", instruction)]
prompt_template_fn, full_prompt = llama_prompt.get_template(messages)
print(full_prompt)

from llm_retrieval_qa.pipeline.chain import QAChain

qa_chain = QAChain(llm, faiss_db, prompt_template_fn, top_k=10, return_source_documents=True, similarity_score_threshold=None)


print("=================start=====================")
question = "How to build a TensorRT engine?"
res = qa_chain(question)
print(f"Question: {res['query']}\nAnswer: {res['result']}")
print("=================")
