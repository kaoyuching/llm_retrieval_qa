import sys
sys.path.append("./")
import warnings
warnings.filterwarnings("ignore")

from langchain_core.documents import Document
from langchain_text_splitters import HTMLSectionSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings

from configs import settings, model_config, get_prompt_template
from splitter import split_html
import llm_prompt

from model_loader import load_model


def load_doc_data(settings):
    with open(settings.doc_file_name, 'r', encoding='utf-8') as f:
        data = f.read()
    return data

html_doc = load_doc_data(settings)

embedding_model_path = model_config["embedding_model_path"]
llm_model_runtime_kwargs = model_config["runtime"]


headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
    ("h4", "Header 4"),
    ("h5", "Header 5"),
    ("table", 'table'),
]

chunk_size = 1000 # 800
chunk_overlap = 50 # 30
splits = split_html(
    html_doc,
    encoding='utf-8',
    sections_to_split=headers_to_split_on,
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", ",", "."],
)


# vector store
hf_embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_path,
    encode_kwargs={'normalize_embeddings': True},
)

faiss_db = FAISS.from_documents(
    splits,
    hf_embeddings,
    distance_strategy=DistanceStrategy.COSINE,
)

retriever = faiss_db.as_retriever(
    # search_type='similarity',  # mmr
    search_type='mmr',  # mmr
    search_kwarg={'k': 20},
)


# prepare prompt
# prompt_template_fn, full_prompt = configs.get_prompt_template()
prompt_template_fn, full_prompt_template = get_prompt_template(model_config["prompt_template"])
print("prompt:\n", full_prompt_template)


model = load_model(model_config, settings.quantization, settings.device)
if model_config["format"] == "hf":
    import torch
    from transformers import AutoTokenizer, pipeline
    from langchain_huggingface import HuggingFacePipeline
    from utils import QAChain

    tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        **llm_model_runtime_kwargs,
    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    qa_chain = QAChain(llm, faiss_db, prompt_template_fn, top_k=10, return_source_documents=True, similarity_score_threshold=None)
elif model_config["format"] == "gguf":
    from utils import QAChainCPP

    qa_chain = QAChainCPP(
        model,
        faiss_db,
        prompt_template_fn,
        top_k=10,
        return_source_documents=True,
        similarity_score_threshold=None,
        model_kwargs=llm_model_runtime_kwargs,
    )
else:
    raise ValueError("model format must be one of ('hf', 'gguf')")


# example questions
from example_data_config import example_questions

print("start answering example questions...")
for question in example_questions:
    print(f"Question: {question}")
    res = qa_chain(question)
    # print(f"Question: {res['query']}\nAnswer: {res['result']}")
    print(f"Answer: {res['result']}")
    print("=================")
