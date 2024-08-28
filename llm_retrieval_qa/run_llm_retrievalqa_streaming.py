import os
import sys
sys.path.append("./")
import warnings
warnings.filterwarnings("ignore")

from llm_retrieval_qa.configs import settings, model_config, vector_store_config,\
    get_prompt_template, get_embedding_fn
from llm_retrieval_qa.splitter import split_html
from llm_retrieval_qa.pipeline.model_loader import load_model


def load_doc_data(settings):
    with open(settings.doc_file_name, 'r', encoding='utf-8') as f:
        data = f.read()
    return data

doc_fname = os.path.basename(settings.doc_file_name)
html_doc = load_doc_data(settings)

llm_model_runtime_kwargs = model_config["runtime"]


headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
    ("h4", "Header 4"),
    ("h5", "Header 5"),
    ("table", 'table'),
]

chunk_size = 500 # 1000
chunk_overlap = 30 # 50
splits = split_html(
    html_doc,
    encoding='utf-8',
    sections_to_split=headers_to_split_on,
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", ",", "."],
)


embedding_fn = get_embedding_fn(model_config["embedding_cfgs"])


# vector store
if vector_store_config.type == "milvus":
    from llm_retrieval_qa.vector_store.milvus import DbMilvus

    vector_db = DbMilvus(
        embedding_fn,
        vector_store_config.uri,
        db_name=vector_store_config.db_name,
        collection_name=vector_store_config.collection,
    )

    texts = [x.dict()["page_content"] for x in splits]
    # check if the doc is already in the vector db using doc_fname
    exist_docs = vector_db.get(f'doc_fname == "{doc_fname}"')
    if len(exist_docs) == 0:
        _ = vector_db.create(texts, doc_fname=doc_fname)
elif vector_store_config.type == "faiss":
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy

    vector_db = FAISS.from_documents(
        splits,
        embedding_fn,
        distance_strategy=DistanceStrategy.COSINE,
    )

    retriever = vector_db.as_retriever(
        # search_type='similarity',  # mmr
        search_type='mmr',  # mmr
        search_kwarg={'k': 20},
    )
else:
    raise ValueError("Invallid vector database")


# prepare prompt
prompt_template_fn, full_prompt_template = get_prompt_template(model_config["prompt_template"])
print("prompt:\n", full_prompt_template)


model = load_model(model_config, settings.quantization, settings.device)
top_k = settings.search_topk

if model_config["format"] == "hf":
    from transformers import AutoTokenizer
    from llm_retrieval_qa.pipeline.streaming import QAHFStreamer, generate_response


    tokenizer = AutoTokenizer.from_pretrained(model_config["model_path"])

    streamer_config = {
        'args': (tokenizer,),
        'kwargs': model_config["streamer"]
    }

    if 'return_full_text' in llm_model_runtime_kwargs:
        llm_model_runtime_kwargs.pop('return_full_text')

    model_kwargs = dict(
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        **llm_model_runtime_kwargs,
    )

    qa_streaming = QAHFStreamer(
        tokenizer,
        model,
        streamer_config,
        vector_db,
        prompt_template_fn,
        top_k=top_k,
        return_source_documents=False,
        similarity_score_threshold=None,
        model_kwargs=model_kwargs,
        device=settings.device,
    )
elif model_config["format"] == "gguf":
    from llm_retrieval_qa.pipeline.streaming import LlamaCppStreamer, generate_response

    qa_streaming = LlamaCppStreamer(
        model,
        vector_db,
        prompt_template_fn,
        streamer_cfg=model_config["streamer"],
        top_k=top_k,
        return_source_documents=False,
        similarity_score_threshold=None,
        model_kwargs=model_config["generate"],
    )


# run examples
example_fname = settings.example_question_file
if example_fname:
    with open(example_fname, "r") as f:
        example_questions = [line.rstrip("\n") for line in f]

    print("start answering example questions...")
    for question in example_questions:
        print(f"Question: {question}")
        print("Answer:")
        qa_streaming(question)
        for x in generate_response(qa_streaming.streamer):
            print(x, end="", flush=True)
        print('')
else:
    while True:
        question = input("Enter Question:")
        if question.lower() in ["exit", "quit"]:
            break
        print("Answer:")
        qa_streaming(question)
        for x in generate_response(qa_streaming.streamer):
            print(x, end="", flush=True)
        print('')
