import sys
sys.path.append("./")
import onnx
import onnxruntime
import torch
from transformers import AutoConfig, AutoTokenizer, TextStreamer
from optimum.onnxruntime import ORTModelForCausalLM

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from splitter import split_html
from llm_prompt import PromptLlama2, PromptLlama3
from utils import similarity_search, get_qa_prompt


r"""
reference:
    - https://mlops.community/mlops-more-oops-than-ops/
"""


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
)

faiss_db = FAISS.from_documents(
    splits,
    hf_embeddings,
    distance_strategy=DistanceStrategy.COSINE,
)


# load tokenizer and onnx model
llm_model_name = "../models/Llama-2-7b-chat-hf-fp16"
onnx_model_dir = "../models/Llama-2-7b-chat-hf-fp16"
config = AutoConfig.from_pretrained(llm_model_name, trust_remote_code=True)
# config.save_pretrained(onnx_model_dir)  # Save config file in ONNX model directory
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
tokenizer.pad_token = "[PAD]"


# llama2 prompt
llama_prompt = PromptLlama2()
system = """You serve as a assistant specialized in answering questions with the given context.
If the following context is not directly related to the question, you must say that you don't know.
Don't try to make up any answers. No potential connection and no guessing."""

instruction = """Base on the following context: {context}, please answer {question}.
If the question is not directly related to the description, you should answer you don't know."""

llama_prompt.set_system_prompt(system_prompt=system)
prompt_template_fn, full_prompt = llama_prompt.get_template(instruction)

questions = [
    "How to say 'thank you' in germany?",
    # "Can you introduce some german food?",
    # "What is and_ in SQLAlchemy?",
    # "What is the compute capability?",
    # "What is TensorRT?",
    # "How to bulid TensorRT engine?",
    # "How to bulid TensorRT engine with python?",
    # "How to bulid TensorRT engine with python from onnx model?",
]
prompt = []
for question in questions:
    doc_str, contexts, scores = similarity_search(faiss_db, question, top_k=10, threshold=None)
    input_prompt = get_qa_prompt(prompt_template_fn, question, contexts)
    prompt.append(input_prompt)
prompt_lens = [len(x) for x in prompt]


device_id = 0
# device = torch.device("cpu")  # Change to torch.device("cpu") if running on CPU
device = torch.device(f"cuda:{device_id}")

# ep = "CPUExecutionProvider"  # change to CPUExecutionProvider if running on CPU
ep = "CUDAExecutionProvider"
ep_options = {"device_id": device_id}

model = ORTModelForCausalLM.from_pretrained(
    onnx_model_dir,
    use_io_binding=False,
    use_cache=False,
    provider=ep,
    # provider_options={"device_id": device_id}  # comment out if running on CPU
)
# inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(torch.float16).to(device)

print("-------------")
# generate_ids = model.generate(**inputs, do_sample=False, max_new_tokens=2048)
for ques, _prompt in zip(questions, prompt):
    prompt_len = len(_prompt) - 1
    inputs = tokenizer(_prompt, return_tensors="pt", padding=True).to(device)
    generate_ids = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=2048,
        num_beams=1,
        temperature=0.0,
        repetition_penalty=1.1,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    transcription = tokenizer.decode(generate_ids[0], skip_special_tokens=True)
    transcription = transcription[prompt_len:]
    print(f"Q: {ques}")
    print(f"A: {transcription}")
    print("-------------")
