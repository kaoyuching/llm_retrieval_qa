import abc
from typing import Optional, List, Dict
from collections import namedtuple
import re
from langchain.prompts import PromptTemplate


Message = namedtuple("Message", "role content")


class PromptBase(abc.ABC):
    def __init__(self, system_prompt: Optional[str] = None):
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

    prompt example with assistant:
    <s>[INST] <<SYS>>
    {{ system_prompt }}
    <</SYS>>

    {{ user_message_1 }} [/INST] {{ model_answer_1 }} </s>
    <s>[INST] {{ user_message_2 }} [/INST]
    '''
    B_SENT = "<s>"
    E_SENT = " </s>\n"
    B_INST = "[INST] "
    E_INST = " [/INST]"
    B_SYS = "<<SYS>>\n" 
    E_SYS = "\n<</SYS>>\n\n"

    def __init__(self, system_prompt: Optional[str] = None):
        self.system_prompt = "You are a helpful, respectful and honest assistant."
        self.set_system_prompt(system_prompt)

    def set_system_prompt(self, system_prompt: Optional[str] = None):
        if system_prompt is not None:
            self.system_prompt = self.B_SYS + system_prompt + self.E_SYS
        else:
            self.system_prompt = self.B_SYS + self.system_prompt + self.E_SYS

    def get_template(self, messages: List[Message], add_bos: bool = True, **kwargs):
        if len(messages) == 0:
            raise ValueError("messages cannot be empty.")

        B_SENT = self.B_SENT if add_bos else ''
        full_prompt_template = B_SENT + self.B_INST + self.system_prompt + messages[0].content + self.E_INST

        if len(messages) > 2:
            for i, message in enumerate(messages[1:]):
                if message.role == "user":
                    full_prompt_template += self.B_SENT + self.B_INST + message.content + self.E_INST
                elif message.role == "assistant":
                    full_prompt_template += ' ' + message.content + self.E_SENT

        input_variables = re.findall('{(\w+)}', full_prompt_template)

        prompt_template = PromptTemplate(
            input_variables=input_variables,
            template=full_prompt_template,
        )
        return prompt_template, full_prompt_template


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
    B_TEXT = "<|begin_of_text|>"
    B_ROLE = "<|start_header_id|>"
    E_ROLE = "<|end_header_id|>\n\n"
    E_INPUT = "<|eot_id|>\n"

    def __init__(self, system_prompt: Optional[str] = None):
        self.system_prompt = "You are a helpful AI assistant"
        self.set_system_prompt(system_prompt)

    def set_system_prompt(self, system_prompt: Optional[str] = None):
        if system_prompt is not None:
            self.system_prompt = self.B_ROLE + "system" + \
                self.E_ROLE + system_prompt + self.E_INPUT
        else:
            self.system_prompt = self.B_ROLE + "system" + \
                self.E_ROLE + self.system_prompt + self.E_INPUT

    def get_template(self, messages: List[Message], add_bos: bool = True, **kwargs):
        B_TEXT = self.B_TEXT if add_bos else ''
        ROLE_USER = self.B_ROLE + "user" + self.E_ROLE
        ROLE_ASST = self.B_ROLE + "assistant" + self.E_ROLE

        full_prompt_template = B_TEXT + self.system_prompt
        for message in messages:
            if message.role == "user":
                full_prompt_template += ROLE_USER + message.content + self.E_INPUT
            elif message.role == "assistant":
                full_prompt_template += ROLE_ASST + message.content + self.E_INPUT
        full_prompt_template += ROLE_ASST

        input_variables = re.findall('{(\w+)}', full_prompt_template)

        prompt_template = PromptTemplate(
            input_variables=input_variables,
            template=full_prompt_template,
        )
        return prompt_template, full_prompt_template


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
    B_SYS = "<|system|>\n"
    B_ROLE = "<|user|>\n"
    B_ASST = "<|assistant|>\n"
    END = "<|end|>\n"

    def __init__(self, system_prompt: Optional[str] = None):
        self.system_prompt = "You are a helpful assistant."
        self.set_system_prompt(system_prompt)

    def set_system_prompt(self, system_prompt: Optional[str] = None):
        if system_prompt is not None:
            self.system_prompt = self.B_SYS + system_prompt + self.END
        else:
            self.system_prompt = self.B_SYS + self.system_prompt + self.END

    def get_template(self, messages: List[Message], **kwargs):
        r"""
        content: [{role: ..., content: ...}]
        """
        full_prompt_template = self.system_prompt + self.B_ASST
        for message in messages:
            if message.role == "user":
                full_prompt_template += self.B_ROLE + message.content + self.END
            elif message.role == "assistant":
                full_prompt_template += self.B_ASST + message.content + self.END
        full_prompt_template += self.B_ASST

        input_variables = re.findall('{(\w+)}', full_prompt_template)

        prompt_template = PromptTemplate(
            input_variables=input_variables,
            template=full_prompt_template,
        )
        return prompt_template, full_prompt_template


def get_qa_prompt(prompt_template, question, contexts):
    if len(contexts) == 0:
        contexts = "Found nothing in the documents"
    else:
        contexts = '\n'.join(contexts)
    input_prompt = prompt_template.format(context=contexts, question=question)
    return input_prompt
