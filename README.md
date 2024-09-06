# LLM Retrieval QA

Demonstrate retrieval QA with multiple documents that come from different file types.

## Environment
- [x] python >= 3.8


## Prepare data
#### Get `html` from URL

Details are in [`html_parser.py`](https://github.com/kaoyuching/llm_retrieval_qa/blob/master/data_parser/html_parser.py)
Simple usage:

```=shell
$ python ./data_parser/html_parser.py [source url] -c html_data_parse_example.json
```

Example data are from:
1. NVIDIA tensorrt developer guide: [https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
2. SQLAlchemy document: [https://docs.sqlalchemy.org/en/20/core/sqlelement.html](https://docs.sqlalchemy.org/en/20/core/sqlelement.html)
3. Learn german: [https://www.studying-in-germany.org/learn-german/](https://www.studying-in-germany.org/learn-german/)

## Acceptable data file formats
- [x] html
- [x] pdf
- [x] txt
- [x] json
- [] doc, docx


## Export model to onnx format

1. Convert with onnxruntime

```=shell
$ python -m onnxruntime.transformers.models.llama.convert_to_onnx -m "../models/Llama-2-7b-chat-hf" --output "../models/Llama-2-7b-chat-hf-fp16" --precision "fp16" --execution_provider "cpu"
```

2. Convert with optimum

```=shell
$ optimum-cli export onnx --model "../models/Llama-2-7b-chat-hf" --task text-generation --opset 16 --device cuda --dtype fp16 "../models/Llama-2-7b-chat-hf-fp16"
```


## Convert huggingface model to GGUF
Use `convert_hf_to_gguf.py` in `llama.cpp` to export gguf model. It requires `torch ~= 2.2.1`.

```=shell
$ git clone https://github.com/ggerganov/llama.cpp/tree/master
$ cd llama.cpp
$ python convert_hf_to_gguf.py --outtype q8_0 --use-temp-file --outfile "[output GGUF model dir]" "[huggingface model dir]"
```

llama-cpp-python API document: https://llama-cpp-python.readthedocs.io/en/latest/api-reference/  
Llama2 GGUF models can get from: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF  
Llama3 GGUF models can get from: https://huggingface.co/doriskao/Llama-3-8B-Instruct-GGUF  
Llama3.1 GGUF models can get from: https://huggingface.co/doriskao/Meta-Llama-3.1-8B-Instruct-GGUF


## Vector store
There are many ways to store embedded data and perform vector search. In this repository, I use FAISS / Milvus to store vectors.

Run Milvus in Docker:

```=shell
$ docker compose -f ./milvus-docker-compose.yml up -d
```


## Example pipeline with `huggingface transformers` and `langchain`
Demo models: Llama2, Llama3, Llama3.1(need transformers >= 4.43.3), Phi3-mini

See [`./examples/example_huggingface.py`](https://github.com/kaoyuching/llm_retrieval_qa/blob/master/examples/example_huggingface.py)


## Example pipeline with `llama-cpp-python`
Waring: llama gguf model runs on CPU spends a lot of time (few minutes compare to GPU).  

Demo models: Llama2, Llama3, Llama3.1, Phi3-Mini

Example pipeline using `langchain`: [`./examples/example_gguf_langchain.py`](https://github.com/kaoyuching/llm_retrieval_qa/blob/master/examples/example_gguf_langchain.py)  
Example pipeline using `llama_cpp_python`: [`./examples/example_gguf_llamacpp.py`](https://github.com/kaoyuching/llm_retrieval_qa/blob/master/examples/example_gguf_llamacpp.py)  
llama_cpp will consume all the available CPUs, you can run with the following command on Linux to limit the CPU usage:

```=shell
$ taskset --cpu-list [numerical list of processors] command
```


## Run with `run_llm_retrievalqa.py`
1. Setup `.env` file. Here is the example (embedding using onnx model, language model using gguf model):

```
DOC_FILE_NAME="./example_files/nvidia_doc.html"

VECTOR_STORE__TYPE="milvus"
VECTOR_STORE__URI="http://localhost:19530"
VECTOR_STORE__DB_NAME="docs_db"
VECTOR_STORE__COLLECTION="data_collection"

#VECTOR_STORE__TYPE="faiss"
#VECTOR_STORE__URI="faiss_db.pkl"

EMBEDDING_MODEL_TYPE="onnx"
MODEL_NAME="llama3.1-gguf"
QUANTIZATION="false"
DEVICE="cuda"
SEARCH_TOPK=10
TIMER="true"

EXAMPLE_QUESTION_FILE="./example_questions.txt"
```

2. Run file `./llm_retrieval_qa/run_llm_retrievalqa.py`


## Demo
1. Setup `.env`
2. Run server

```=shell
$ uvicorn api.main:app --host 0.0.0.0 --port 8005
```
