

embedding_type_config = {
    "llama2": {
        "onnx": {
            "__class_name__": "OnnxEmbedding",
            "model_path": "../models/text2vec-large-chinese-onnx/model.onnx",
            "tokenizer_path": "../models/text2vec-large-chinese-onnx/tokenizer.json",
            "tokenizer_cfg_path": "../models/text2vec-large-chinese-onnx/tokenizer_config.json",
        },
        "hf": {
            "__class_name__": "HFEmbedding",
            "model_path": "GanymedeNil/text2vec-large-chinese",
        },
    },
    "llama3": {
        "onnx": {
            "__class_name__": "OnnxEmbedding",
            "model_path": "../models/text2vec-large-chinese-onnx/model.onnx",
            "tokenizer_path": "../models/text2vec-large-chinese-onnx/tokenizer.json",
            "tokenizer_cfg_path": "../models/text2vec-large-chinese-onnx/tokenizer_config.json",
        },
        "hf": {
            "__class_name__": "HFEmbedding",
            "model_path": "GanymedeNil/text2vec-large-chinese",
        },
    },
    "phi3": {
        "onnx": {
            "__class_name__": "OnnxEmbedding",
            "model_path": "../models/text2vec-large-chinese-onnx/model.onnx",
            "tokenizer_path": "../models/text2vec-large-chinese-onnx/tokenizer.json",
            "tokenizer_cfg_path": "../models/text2vec-large-chinese-onnx/tokenizer_config.json",
        },
        "hf": {
            "__class_name__": "HFEmbedding",
            "model_path": "GanymedeNil/text2vec-large-chinese",
        },
    }
}


model_type_config = {
    "llama2": {
        "prompt_template": {
            "__class_name__": "PromptLlama2",
            "system": """You serve as a assistant specialized in answering questions with the given context.
            If the following context is not directly related to the question, you must say that you don't know.
            Don't try to make up any answers. No potential connection and no guessing.""",
            "messages": [
                {"role": "user", "content": """Base on the following context: {context}, please answer {question}.
            If the question is not directly related to the description, you should answer you don't know."""},
            ],
        },
    },
    "llama3": {
        "prompt_template": {
            "__class_name__": "PromptLlama3",
            "system": """Use the following context to answer the user's question.
    If you don't know the answer or the question is not directly related to the context, you should answer you don't know and don't generate any answers.""",
            "messages": [
                {"role": "user", "content": """Please answer this: {question} directly according to the context: {context}"""}
            ]
        }
    },
    "phi3": {
        "prompt_template": {
            "__class_name__": "PromptPhi3",
            "system": """Use the following context to answer the user's question.
        If you don't know the answer or the question is not directly related to the context, you should answer you don't know and don't generate any answers.""",
            "messages": [
                {"role": "user", "content": """Please answer this: {question} directly according to the context: {context}"""}
            ]
        }
    },
}


model_configs_mapping = {
    "llama2-hf": {
        "format": "hf",
        "model_path": "../models/llama2-7b-chat",
        "model_type": "llama2",
        "quantization": "8bit",
        "init": {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        },
        "runtime": {
            "num_beams": 1,
            "max_new_tokens": 2048,
            "num_return_sequences": 1,
            "temperature": 0.0,
            "do_sample": False,
            "repetition_penalty": 1.1,
            "return_full_text": False,
        },
        "streamer": {
            "skip_prompt": True,
            "skip_special_tokens": True,
            "trust_remote_code": True,
        },
        "prompt_kwargs": {"add_bos": True},
    },
    "llama2-gguf": {
        "format": "gguf",
        "model_path": "../models/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q6_K.gguf",
        "model_type": "llama2",
        "init": {
            "n_ctx": 4096,
            "n_threads": 5,
            "verbose": False,
        },
        "runtime": {
            "temperature": 0.0,
            "max_tokens": 2048,
            "top_p": 1,
            "repeat_penalty": 1.1,
            "echo": False,
            "stop": ['</s>'],
        },
        "generate": {
            "temp": 0.0,
            "top_p": 1,
            "repeat_penalty": 1.1,
            "max_tokens": 2048,
            "stop": ['</s>'],
        },
        "streamer": {'timeout': None},
        "prompt_kwargs": {"add_bos": True},
    },
    "llama3-hf": {
        "format": "hf",
        "model_path": "../models/Meta-Llama-3-8B-Instruct",
        "model_type": "llama3",
        "quantization": "8bit",
        "init": {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        },
        "runtime": {
            "num_beams": 1,
            "max_new_tokens": 2048,
            "num_return_sequences": 1,
            "temperature": 0.0,
            "do_sample": False,
            "repetition_penalty": 1.1,
            "top_p": 1,
            "return_full_text": False,
        },
        "streamer": {
            "skip_prompt": True,
            "skip_special_tokens": True,
            "trust_remote_code": True,
        },
        "prompt_kwargs": {"add_bos": True},
    },
    "llama3-gguf": {
        "format": "gguf",
        "model_path": "../models/Llama-3-8B-Instruct-GGUF/Llama-3-8B-Instruct-Q8_0.gguf",
        "model_type": "llama3",
        "init": {
            "n_ctx": 8192,
            "n_threads": 5,
            "verbose": False,
        },
        "runtime": {
            "temperature": 0.0,
            "max_tokens": 2048,
            "top_p": 1,
            "repeat_penalty": 1.1,
            "echo": False,
            "stop": ['<|eot_id|>', '<|end_of_text|>'],
        },
        "generate": {
            "temp": 0.0,
            "top_p": 1,
            "repeat_penalty": 1.1,
            "max_tokens": 2048,
            "stop": ['<|eot_id|>', '<|end_of_text|>'],
        },
        "streamer": {'timeout': None},
        "prompt_kwargs": {"add_bos": True},
    },
    "llama3.1-hf": {
        "format": "hf",
        "model_path": "../models/Meta-Llama-3.1-8B-Instruct",
        "model_type": "llama3",
        "quantization": "8bit",
        "init": {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        },
        "runtime": {
            "num_beams": 1,
            "max_new_tokens": 2048,
            "num_return_sequences": 1,
            "temperature": 0.0,
            "do_sample": False,
            "repetition_penalty": 1.1,
            "return_full_text": False,
        },
        "streamer": {
            "skip_prompt": True,
            "skip_special_tokens": True,
            "trust_remote_code": True,
        },
        "prompt_template": {"add_bos": True},
    },
    "llama3.1-gguf": {
        "format": "gguf",
        "model_path": "../models/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
        "model_type": "llama3",
        "init": {
            "n_ctx": 12288,
            "n_threads": 5,
            "verbose": False,
        },
        "runtime": {
            "temperature": 0.0,
            "max_tokens": 2048,
            "top_p": 1,
            "repeat_penalty": 1.1,
            "echo": False,
            "stop": ['<|eot_id|>', '<|end_of_text|>'],
        },
        "generate": {
            "temp": 0.0,
            "top_p": 1,
            "repeat_penalty": 1.1,
            "max_tokens": 2048,
            "stop": ['<|eot_id|>', '<|end_of_text|>'],
        },
        "streamer": {"timeout": None},
        "prompt_template": {"add_bos": True},
    },
    "phi3-mini-4k-hf": {
        "format": "hf",
        "model_path": "microsoft/Phi-3-mini-4k-instruct",
        "model_type": "phi3",
        "quantization": "8bit",
        "init": {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        },
        "runtime": {
            "num_beams": 1,
            "max_new_tokens": 2048,
            "num_return_sequences": 1,
            "temperature": 0.0,
            "do_sample": False,
            "repetition_penalty": 1.1,
            "return_full_text": False,
        },
        "streamer": {
            "skip_prompt": True,
            "skip_special_tokens": True,
            "trust_remote_code": True,
        },
    },
    "phi3-mini-4k-gguf": {
        "format": "gguf",
        "model_path": "../models/Phi-3-mini-4k-instruct-gguf/Phi-3-mini-4k-instruct-q4.gguf",
        "model_type": "phi3",
        "init": {
            "n_ctx": 12288,
            "n_threads": 5,
            "verbose": False,
        },
        "runtime": {
            "temperature": 0.0,
            "max_tokens": 2048,
            "top_p": 1,
            "repeat_penalty": 1.1,
            "echo": False,
            "stop": ['<|end|>', '<|endoftext|>'],
        },
        "generate": {
            "temp": 0.0,
            "top_p": 1,
            "repeat_penalty": 1.1,
            "max_tokens": 2048,
            "stop": ['<|end|>', '<|endoftext|>'],
        },
        "streamer": {"timeout": None}
    },
}


def get_model_config(model_name):
    config = model_configs_mapping[model_name]
    model_type = config['model_type']
    config['prompt_template'] = {**model_type_config[model_type]["prompt_template"], **config.get('prompt_template', {})}
    config['embedding_cfgs'] = {**embedding_type_config[model_type]}
    return config
