import sys
sys.path.append("./")
import os
import warnings
warnings.filterwarnings("ignore")

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from splitter import split_html
from llm_prompt import PromptLlama2, PromptLlama3
from utils import similarity_search, get_qa_prompt

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
# filename = './example_files/sql_alchemy_doc_all.html'
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


# llama2 prompt
# llama_prompt = PromptLlama2()
# system = """You serve as a assistant specialized in answering questions with the given context.
# If the following context is not directly related to the question, you must say that you don't know.
# Don't try to make up any answers. No potential connection and no guessing."""

# instruction = """Base on the following context: {context}, please answer {question}.
# If the question is not directly related to the description, you should answer you don't know."""

# llama3 prompt
llama_prompt = PromptLlama3()
system = """Use the following context to answer the user's question.
If you don't know the answer or the question is not directly related to the context, you should answer you don't know and don't generate any answers."""

instruction = """Please answer the {question} directly according to the context: {context}"""

llama_prompt.set_system_prompt(system_prompt=system)
prompt_template_fn, full_prompt = llama_prompt.get_template(instruction)


# gguf model
r'''
# use with langchain
llm = LlamaCpp(
    # model_path="../models/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q6_K.gguf",
    model_path="../models/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf",
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
'''

# with llama_cpp
# document: https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__init__
from llama_cpp import Llama
llm_cpp = Llama(
    # model_path="../models/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q6_K.gguf",
    model_path="../models/Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q8_0.gguf",
    n_ctx=4096,  # llama2 context window
    n_threads=10,
    verbose=False,
)

llm_cpp_kwargs = dict(
    temperature=0.0,
    max_tokens=2048,
    top_p=1,
    repeat_penalty=1.1,
    echo=False,
)


from utils import QAChainCPP

qa_chain_cpp = QAChainCPP(
    llm_cpp,
    faiss_db,
    prompt_template_fn,
    top_k=10,
    return_source_documents=True,
    similarity_score_threshold=None,
    model_kwargs=llm_cpp_kwargs,
)


print("start answering question...")
question = "How to say 'thank you' in germany?"
res = qa_chain_cpp(question)
print(f"Question: {res['query']}\nAnswer: {res['result']}")
print("=================")


question = "Can you introduce some german food?"
res = qa_chain_cpp(question)
print(f"Question: {res['query']}\nAnswer: {res['result']}")
print("=================")


question = "What is and_ in SQLAlchemy?"
res = qa_chain_cpp(question)
print(f"Question: {res['query']}\nAnswer: {res['result']}")
print("=================")


question = "How to use colume with SQLAlchemy?"
res = qa_chain_cpp(question)
print(f"Question: {res['query']}\nAnswer: {res['result']}")
print("=================")


question = "What is the compute capability?"
res = qa_chain_cpp(question)
print(f"Question: {res['query']}\nAnswer: {res['result']}")
print("=================")


question = "What is TensorRT?"
res = qa_chain_cpp(question)
print(f"Question: {res['query']}\nAnswer: {res['result']}")
print("=================")


question = "Can you explain what is TensorRT?"
res = qa_chain_cpp(question)
print(f"Question: {res['query']}\nAnswer: {res['result']}")
print("=================")


question = "How to bulid TensorRT engine?"
res = qa_chain_cpp(question)
print(f"Question: {res['query']}\nAnswer: {res['result']}")
print("=================")


question = "How to bulid TensorRT engine with python?"
res = qa_chain_cpp(question)
print(f"Question: {res['query']}\nAnswer: {res['result']}")
print("=================")


question = "How to bulid TensorRT engine with python from onnx model?"
res = qa_chain_cpp(question)
print(f"Question: {res['query']}\nAnswer: {res['result']}")
print("=================")
