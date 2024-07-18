import sys
sys.path.append("./")
import os
# from threadpoolctl import threadpool_limits
import warnings
warnings.filterwarnings("ignore")

import traceback

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from splitter import split_html
from llm_prompt import PromptLlama2, PromptLlama3
from utils import similarity_search, get_qa_prompt


# https://blog.infuseai.io/llama-2-llama-cpp-python-introduction-c5f67d979eaa
# llama2 7b ggml: https://huggingface.co/TheBloke/Llama-2-7B-GGML
# llama2 7b gguf (TheBloke/Llama-2-7B-Chat-GGUF): https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF
# llama now goes faster on CPUs: https://justine.lol/matmul/


# load reference dataset
filename = './example_files/sql_alchemy_doc_all.html'
headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
    ("h4", "Header 4"),
    ("h5", "Header 5"),
    ("table", 'table'),
]
splits = split_html(
    filename,
    encoding='utf-8',
    sections_to_split=headers_to_split_on,
    chunk_size=1000,
    chunk_overlap=50,
    separators=["\n\n", "\n", ",", "."],
)


# vector database
model_name = "GanymedeNil/text2vec-large-chinese"
hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    encode_kwargs={'normalize_embeddings': True},
    # model_kwargs={"device": "cpu"},
)

faiss_db = FAISS.from_documents(
    splits,
    hf_embeddings,
    distance_strategy=DistanceStrategy.COSINE,
)


# llama2 prompt
llama_prompt = PromptLlama2()
system = """You serve as a assistant specialized in answering questions with the given context.
If the following context is not directly related to the question, you must say that you don't know.
Don't try to make up any answers. No potential connection and no guessing."""

instruction = """Base on the following context: {context}, please answer {question}.
If the question is not directly related to the description, you should answer you don't know."""

llama_prompt.set_system_prompt(system_prompt=system)
prompt_template_fn, full_prompt = llama_prompt.get_template(instruction)


# gguf model
llm = LlamaCpp(
    model_path="../models/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q6_K.gguf",
    add_space_prefix=False,
    temperature=0.0,
    n_ctx=4096,  # llama2 context window
    max_tokens=2048,
    top_p=1,
    repeat_penalty=1.1,
    # n_gpu_layers=-1,  # use with gpu
    n_threads=30,
    # streaming=True,
    verbose=False,
)


import time

print("start answering question...")
st = time.time()
question = "How to say 'thank you' in germany?"
doc_str, contexts, scores = similarity_search(faiss_db, question, top_k=10, threshold=None)
input_prompt = get_qa_prompt(prompt_template_fn, question, contexts)
print(f"Question: {question}")
res = llm.invoke(input_prompt)
print("time:", time.time() - st)
print(res)


st = time.time()
question = "What is and_ in SQLAlchemy?"
doc_str, contexts, scores = similarity_search(faiss_db, question, top_k=10, threshold=None)
input_prompt = get_qa_prompt(prompt_template_fn, question, contexts)
print(f"Question: {question}")
res = llm.invoke(input_prompt)
print("time:", time.time() - st)
print(res)
