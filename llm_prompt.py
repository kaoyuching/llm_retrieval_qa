import abc
from typing import Optional
import re
from langchain.prompts import PromptTemplate


class PromptBase(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def set_system_prompt(self, system_prompt: Optional[str] = None):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_template(self, instruction: str, **kwargs):
        raise NotImplementedError()


class PromptLlama2(PromptBase):
    r'''
    PromptLlama2 is used by Llama2.

    llama2: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-2
    llama2 prompt template: https://github.com/meta-llama/llama/blob/main/llama/generation.py#L284-L395

    prompt example:
    <s>[INST] <<SYS>>
    {{ system_prompt }}
    <</SYS>>

    {{ user_message }} [/INST]
    '''
    def __init__(self):
        self.B_SENT = "<s>"
        self.E_SENT = "</s>"
        self.B_INST = "[INST]"
        self.E_INST = "[/INST]"
        self.B_SYS = "<<SYS>>\n" 
        self.E_SYS = "\n<</SYS>>\n\n"
        self.system_prompt = """You are a helpful, respectful and honest assistant.
        Always answer as helpfully as possible, while being safe.
        Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
        Please ensure that your responses are socially unbiased and positive in nature.
        If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.
        If you don't know the answer to a question, please don't share false information."""

    def set_system_prompt(self, system_prompt: Optional[str] = None):
        if system_prompt is not None:
            self.system_prompt = self.B_SYS + system_prompt + self.E_SYS
        else:
            self.system_prompt = self.B_SYS + self.system_prompt + self.E_SYS

    def get_template(self, instruction: str, add_bos: bool = True, **kwargs):
        B_SENT = self.B_SENT if add_bos else ''
        full_prompt = B_SENT + self.B_INST + ' ' + self.system_prompt + instruction + ' ' + self.E_INST
        input_variables = re.findall('{(\w+)}', instruction)

        prompt_template = PromptTemplate(
            input_variables=input_variables,
            template=full_prompt,
        )
        return prompt_template, full_prompt


class PromptLlama3(PromptBase):
    r'''
    PromptLlama3 is used by Llama3 and Llama3.1.

    llama3: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3
    llama3 template: https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py#L203

    prompt example:
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are a helpful AI assistant for travel tips and recommendations<|eot_id|>
    <|start_header_id|>user<|end_header_id|>

    What can you help me with?<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    '''
    def __init__(self):
        self.B_TEXT = "<|begin_of_text|>"
        self.B_ROLE = "<|start_header_id|>"
        self.E_ROLE = "<|end_header_id|>\n\n"
        self.E_INPUT = "<|eot_id|>\n"
        self.system_prompt = """You are a helpful AI assistant"""

    def set_system_prompt(self, system_prompt: Optional[str] = None):
        if system_prompt is not None:
            self.system_prompt = self.B_ROLE + "system" + \
                self.E_ROLE + system_prompt + self.E_INPUT
        else:
            self.system_prompt = self.B_ROLE + "system" + \
                self.E_ROLE + self.system_prompt + self.E_INPUT

    def get_template(self, instruction: str, add_bos: bool = True, **kwargs):
        B_TEXT = self.B_TEXT if add_bos else ''
        full_prompt = B_TEXT + self.system_prompt + self.B_ROLE + "user" + self.E_ROLE +\
                instruction + self.E_INPUT + self.B_ROLE + "assistant" + self.E_ROLE

        input_variables = re.findall('{(\w+)}', instruction)

        prompt_template = PromptTemplate(
            input_variables=input_variables,
            template=full_prompt,
        )
        return prompt_template, full_prompt


class PromptPhi3(PromptBase):
    r"""
    PromptPhi3 is used by Microsoft Phi3.

    Microsoft phi3: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct

    prompt example:
        <|system|>
        You are a helpful assistant.<|end|>
        <|user|>
        Question?<|end|>
        <|assistant|>
    """
    def __init__(self):
        self.B_SYS = "<|system|>\n"
        self.B_ROLE = "<|user|>\n"
        self.B_ASST = "<|assistant|>\n"
        self.END = "<|end|>\n"
        self.system_prompt = "You are a helpful assistant."

    def set_system_prompt(self, system_prompt: Optional[str] = None):
        if system_prompt is not None:
            self.system_prompt = self.B_SYS + system_prompt + self.END
        else:
            self.system_prompt = self.B_SYS + self.system_prompt + self.END

    def get_template(self, instruction: str, **kwargs):
        full_prompt = self.system_prompt + self.B_ROLE + instruction + self.END + self.B_ASST

        input_variables = re.findall('{(\w+)}', instruction)

        prompt_template = PromptTemplate(
            input_variables=input_variables,
            template=full_prompt,
        )
        return prompt_template, full_prompt
