import sys
sys.path.append("./")
import os
import warnings
warnings.filterwarnings("ignore")
import atexit

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from llm_retrieval_qa.splitter import split_html
from llm_retrieval_qa.pipeline.llm_prompt import PromptLlama2, PromptLlama3, Message
from llm_retrieval_qa.pipeline.chain import similarity_search, get_qa_prompt

r"""
llama_cpp will consume all CPUs on your device.
If you run on the linux, you can run with the following example command:
[limit 20 CPUs]

taskset --cpu-list 0-20 python ./llm_retrievalqa_gguf.py
"""


# https://blog.infuseai.io/llama-2-llama-cpp-python-introduction-c5f67d979eaa
# llama2 7b ggml: https://huggingface.co/TheBloke/Llama-2-7B-GGML
# llama2 7b gguf (TheBloke/Llama-2-7B-Chat-GGUF): https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF
# llama now goes faster on CPUs: https://justine.lol/matmul/


# load reference dataset
filename = './example_files/nvidia_doc.html'
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


# llama3 prompt
system = """Use the following context to answer the user's question.
If you don't know the answer or the question is not directly related to the context, you should answer you don't know and don't generate any answers."""

instruction = """Please answer the {question} directly according to the context: {context}"""

llama_prompt = PromptLlama3(system)
messages = [Message("user", instruction)]
prompt_template_fn, full_prompt = llama_prompt.get_template(messages)


# gguf model
# use with langchain
model = LlamaCpp(
    model_path="../models/Llama-3-8B-Instruct-GGUF/Llama-3-8B-Instruct-Q8_0.gguf",
    add_space_prefix=False,
    temperature=0.0,
    n_ctx=8192,  # llama3 context window
    max_tokens=2048,
    top_p=1,
    repeat_penalty=1.1,
    # n_gpu_layers=-1,  # use with gpu
    n_threads=30,
    # streaming=True,
    verbose=False,
)

from llm_retrieval_qa.pipeline.chain import QAChain

qa_chain = QAChain(
    model,
    faiss_db,
    prompt_template_fn,
    top_k=10,
    return_source_documents=True,
    similarity_score_threshold=None,
)

@atexit.register
def free_model():
    model.client.close()


print("start answering question...")
question = "How to build a TensorRT engine?"
res = qa_chain(question)
print(f"Question: {res['query']}\nAnswer: {res['result']}")
print("=================")
