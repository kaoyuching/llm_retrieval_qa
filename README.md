## LLM Retrieval QA

Demonstrate retrieval QA with multiple documents that come from different file types.

### Environment
- [x] python >= 3.8

### Prepare data
#### Get `html` from URL

Details are in [`html_parser.py`](https://github.com/kaoyuching/llm_retrieval_qa/blob/master/data_parser/html_parser.py)
Simple usage:

```=shell
$ python ./data_parser/html_parser.py [source url] -c html_data_parse_example.json
```

Example data:
1. `nvidia_doc.html`: from [https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
2. `sql_alchemy_doc_all.html`: from [https://docs.sqlalchemy.org/en/20/core/sqlelement.html](https://docs.sqlalchemy.org/en/20/core/sqlelement.html)
3. `germany_beginner.html`: from [https://www.studying-in-germany.org/learn-german/](https://www.studying-in-germany.org/learn-german/)

#### Get `pdf` data
TODO

### Export to onnx

1. Convert with onnxruntime

```=shell
$ python -m onnxruntime.transformers.models.llama.convert_to_onnx -m "../models/Llama-2-7b-chat-hf" --output "../models/Llama-2-7b-chat-hf-fp16" --precision "fp16" --execution_provider "cpu"
```

2. Convert with optimum

```=shell
$ optimum-cli export onnx --model "../models/Llama-2-7b-chat-hf" --task text-generation --opset 16 --device cuda --dtype fp16 "../models/Llama-2-7b-chat-hf-fp16"
```


### Example pipeline with `huggingface transformers` and `langchain`
Demo models: Llama2, Llama3

See [`llm_retrievalqa.py`](https://github.com/kaoyuching/llm_retrieval_qa/blob/master/llm_retrievalqa.py)


### Example pipeline with `llama-cpp-python`
Waring: llama gguf model runs on CPU spends a lot of time (few minutes compare to GPU)

Demo models: Llama2, Llama3

#### Convert huggingface model to GGUF
Use `convert_hf_to_gguf.py` in  `llama.cpp` to export gguf model. It requires `torch >= 2.2`.

```=shell
$ git clone https://github.com/ggerganov/llama.cpp/tree/master
$ cd llama.cpp
$ python convert_hf_to_gguf.py --outtype q8_0 --use-temp-file --outfile "[output GGUF model dir]" "[huggingface model dir]"
```


llama-cpp-python API document: https://llama-cpp-python.readthedocs.io/en/latest/api-reference/  
Llama2 GGUF models can get from: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF  

Details are in the file [`llm_retrievalqa_gguf.py`](https://github.com/kaoyuching/llm_retrieval_qa/blob/master/llm_retrievalqa_gguf.py)  
llama_cpp will consume all the available CPUs, you can run with `taskset --cpu-list [numerical list of processors] command` on Linux to limit the CPU usage.
