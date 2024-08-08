import sys
sys.path.append("./")
import os
import warnings
warnings.filterwarnings("ignore")
import atexit

from langchain_community.llms import LlamaCpp

from llm_retrieval_qa.splitter import split_html
from llm_retrieval_qa.vector_store import HFEmbedding, DbMilvus
from llm_retrieval_qa.pipeline.llm_prompt import PromptLlama2, PromptLlama3, Message

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
# filename = './example_files/nvidia_doc.html'
filename = './example_files/sql_alchemy_doc_all.html'
doc_fname = os.path.basename(settings.doc_file_name)
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
    chunk_size=500,
    chunk_overlap=30,
    separators=["\n\n", "\n", ",", "."],
)


# vector store: milvus
model_name = "GanymedeNil/text2vec-large-chinese"
hf_embedding = HFEmbedding(model_name, normalize_embeddings=True)

vector_db = DbMilvus(
    hf_embedding,
    "http://localhost:19530",
    db_name="docs_db",
    collection_name="data_collection"
)


# write vectors to db
texts = [x.dict()["page_content"] for x in splits]
exist_docs = vector_db.get(f'doc_fname == "{doc_fname}"')
if len(exist_docs) == 0:
    _ = vector_db.create(texts, doc_fname=doc_fname)


# llama2 prompt
system = """You serve as a assistant specialized in answering questions with the given context.
If the following context is not directly related to the question, you must say that you don't know.
Don't try to make up any answers. No potential connection and no guessing."""

instruction = """Base on the following context: {context}, please answer {question}.
If the question is not directly related to the description, you should answer you don't know."""

llama_prompt = PromptLlama2(system)
messages = [Message("user", instruction)]
prompt_template_fn, full_prompt = llama_prompt.get_template(messages)


# gguf model
# with llama_cpp
# document: https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__init__
from llama_cpp import Llama
model = Llama(
    model_path="../models/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q6_K.gguf",
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


from llm_retrieval_qa.pipeline.chain import QAChainCPP

qa_chain_cpp = QAChainCPP(
    model,
    vector_db,
    prompt_template_fn,
    top_k=10,
    return_source_documents=True,
    similarity_score_threshold=None,
    model_kwargs=llm_cpp_kwargs,
)

@atexit.register
def free_model():
    model.close()


print("start answering question...")
question = "How to build a TensorRT engine?"
res = qa_chain_cpp(question)
print(f"Question: {res['query']}\nAnswer: {res['result']}")
print("=================")
