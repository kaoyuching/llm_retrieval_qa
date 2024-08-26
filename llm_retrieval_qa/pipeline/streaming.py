from typing import Dict, Optional
from threading import Thread
from langchain.prompts import PromptTemplate
from transformers import TextIteratorStreamer

from llm_retrieval_qa.pipeline.search import similarity_search
from llm_retrieval_qa.pipeline.llm_prompt import get_qa_prompt


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


def generate_response(streamer):
    for text in streamer:
        yield text
