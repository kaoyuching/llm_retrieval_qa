from typing import Dict, Optional, Literal, Union
import copy
from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from llm_retrieval_qa.pipeline import llm_prompt
from llm_retrieval_qa.pipeline.llm_prompt import Message
from llm_retrieval_qa.runtime_config import get_model_config


class VectorStoreSetting(BaseModel):
    type: Literal[str]


class MilvusSetting(VectorStoreSetting):
    type: Literal["milvus"] = "milvus"
    db_name: str
    uri: str
    collection: str


class FAISSSetting(VectorStoreSetting):
    type: Literal["faiss"] = "faiss"
    db_name: str


class Settings(BaseSettings):
    doc_file_name: str
    vector_store: Union[MilvusSetting, FAISSSetting] = Field(..., discriminator="type")
    model_name: str
    quantization: bool = False
    device: str = "cpu"
    search_topk: int = 10
    timer: bool = False
    example_question_file: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
        env_nested_delimiter="__",
    )


def get_prompt_template(prompt_template_config: Dict):
    r"""
    prompt_template_config = {"__class_name__": ..., "system": ..., "messages": ..., **kwargs}
    """
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
vector_store_config = settings.vector_store
