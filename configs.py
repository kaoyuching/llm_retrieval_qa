import sys
sys.path.append("./")

from typing import Dict
import copy
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

import llm_prompt
from llm_prompt import Message
from runtime_config import get_model_config


class Settings(BaseSettings):
    doc_file_name: str
    model_name: str
    quantization: bool = False
    device: str = "cpu"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
    )


def get_prompt_template(prompt_template_config: Dict):
    r"""
    prompt_template_config = {"__class_name__": ..., "system": ..., "messages": ..., **kwargs}
    """
    print(prompt_template_config)
    prompt_template_config["messages"] = [Message(x["role"], x["content"]) for x in prompt_template_config["messages"]]
    prompt_template_kwargs = {
        k: v for k, v in prompt_template_config.items() 
        if k not in ["__class_name__", "system", "messages"]}
    _prompt_template_obj = getattr(llm_prompt, prompt_template_config["__class_name__"])
    prompt_template_obj = _prompt_template_obj(prompt_template_config["system"])
    prompt_template_fn, full_prompt_template = prompt_template_obj.get_template(prompt_template_config["messages"], **prompt_template_kwargs)
    return prompt_template_fn, full_prompt_template


settings = Settings()
# get model config
model_config = get_model_config(settings.model_name)
