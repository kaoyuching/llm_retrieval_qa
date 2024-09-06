from typing import Dict, Optional
from threading import Thread
from queue import Queue
from langchain.prompts import PromptTemplate

from llm_retrieval_qa.pipeline.search import similarity_search
from llm_retrieval_qa.pipeline.llm_prompt import get_qa_prompt


class CustomTextStreamer():
    def __init__(self, timeout: Optional[float] = None):
        self.text_queue = Queue()
        self.stop_signal = None
        self.timeout = timeout

    def put(self, text):
        self.on_finalized_text(text)

    def end(self):
        self.on_finalized_text("", stream_end=True)

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.text_queue.put(text, timeout=self.timeout)
        if stream_end:
            self.text_queue.put(self.stop_signal, timeout=self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            with self.text_queue.mutex:
                self.text_queue.queue.clear()
            raise StopIteration
        else:
            return value


# streamer for huggingface
class QAHFStreamer():
    def __init__(
        self,
        tokenizer,
        llm_model,
        streamer_cfg: Dict,
        vector_db,
        prompt_template: PromptTemplate,
        top_k: int = 10,
        return_source_documents: bool = False,
        similarity_score_threshold: Optional[float] = None,
        model_kwargs: Dict = {'max_new_tokens': 2048},
        device: str = "cpu",
    ):
        from transformers import TextIteratorStreamer

        self.tokenizer = tokenizer
        self.llm_model = llm_model
        self.streamer = TextIteratorStreamer(*streamer_cfg['args'], **streamer_cfg['kwargs'])
        self.vector_db = vector_db
        self.prompt_template = prompt_template
        self.top_k = top_k
        self.return_source_documents = return_source_documents
        self.threshold = similarity_score_threshold
        self.model_kwargs = model_kwargs
        self.device = device

    def __call__(self, question):
        doc_str, contexts, scores = similarity_search(self.vector_db, question, self.top_k, self.threshold)
        input_prompt = get_qa_prompt(self.prompt_template, question, contexts)

        inputs = self.tokenizer([input_prompt], return_tensors="pt").to(self.device)
        model_kwargs = {**inputs, 'streamer': self.streamer, **self.model_kwargs}

        thread = Thread(target=self.llm_model.generate, kwargs=model_kwargs)
        thread.start()


class LlamaCppStreamer():
    def __init__(
        self,
        llm_model,
        vector_db,
        prompt_template: PromptTemplate,
        streamer_cfg: Dict = {'timeout': None},
        top_k: int = 10,
        return_source_documents: bool = False,
        similarity_score_threshold: Optional[float] = None,
        model_kwargs: Dict = {'max_tokens': 2048},
    ):
        self.llm_model = llm_model
        self.streamer = CustomTextStreamer(**streamer_cfg)
        self.vector_db = vector_db
        self.prompt_template = prompt_template
        self.top_k = top_k
        self.return_source_documents = return_source_documents
        self.threshold = similarity_score_threshold
        self.model_kwargs = {**model_kwargs}
        self.max_tokens = self.model_kwargs.pop("max_tokens") if "max_tokens" in self.model_kwargs else 16
        stop = self.model_kwargs.pop("stop") if "stop" in self.model_kwargs else []
        self.stop_tokens = self.llm_model.tokenize(''.join(stop).encode("utf-8"), add_bos=False, special=True)

    def model_generate(self, input_tokens, max_tokens: int = 16, gen_kwargs: Dict = dict()):
        from llama_cpp import llama_cpp

        return_tokens = 0
        for token in self.llm_model.generate(input_tokens, **gen_kwargs):
            if llama_cpp.llama_token_is_eog(self.llm_model._model.model, token) or token in self.stop_tokens or return_tokens >= max_tokens:
                text = self.llm_model.detokenize([], prev_tokens=input_tokens)
                self.streamer.put(text.decode("utf-8", "ignore"))
                break
            text = self.llm_model.detokenize([token], prev_tokens=input_tokens)
            return_tokens += 1
            self.streamer.put(text.decode("utf-8", "ignore"))
        self.streamer.end()

    def __call__(self, question):
        doc_str, contexts, scores = similarity_search(self.vector_db, question, self.top_k, threshold=self.threshold)
        input_prompt = get_qa_prompt(self.prompt_template, question, contexts)

        input_tokens = self.llm_model.tokenize(input_prompt.encode("utf-8"), add_bos=False, special=True)
        model_kwargs = dict(input_tokens=input_tokens, max_tokens=self.max_tokens, gen_kwargs=self.model_kwargs)

        thread = Thread(target=self.model_generate, kwargs=model_kwargs)
        thread.start()


def generate_response(streamer):
    for text in streamer:
        yield text
