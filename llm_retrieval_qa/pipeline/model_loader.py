from typing import Dict
import copy
# load model config
from llm_retrieval_qa.runtime_config import get_model_config


def load_model(
        model_config: Dict,
        quantization: bool = False,
        device: str = 'cpu',
        ) -> Dict[str, Dict]:
    model_path = model_config["model_path"]
    model_format = model_config["format"]
    model_kwargs = copy.deepcopy(model_config["init"])

    if model_format == "gguf":
        from llama_cpp import Llama

        model = Llama(model_path=model_path, **model_kwargs)
        # deal with exit issue
        model._stack.pop_all()
    elif model_format == "hf":
        import torch
        from transformers import AutoModelForCausalLM

        model_kwargs["device_map"] = device
        # quantization
        if quantization:
            from transformers import BitsAndBytesConfig

            if model_config["quantization"] == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                )
            else:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=False,
                )
            model_kwargs["quantization_config"] = quantization_config

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            **model_kwargs,
        )
    return model
