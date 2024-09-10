from typing import Optional, Dict
from langchain.prompts import PromptTemplate

from llm_retrieval_qa.timeit import time_it
from llm_retrieval_qa.pipeline.llm_prompt import get_qa_prompt
from llm_retrieval_qa.pipeline.search import similarity_search


class QAChain():
    def __init__(
        self,
        llm_model,
        vector_db,
        prompt_template: PromptTemplate,
        top_k: int = 10,
        return_source_documents: bool = False,
        similarity_score_threshold: Optional[float] = None,
        reranking: bool = False,
        rerank_topk: int = 5,
    ):
        self.llm_model = llm_model
        self.vector_db = vector_db
        self.prompt_template = prompt_template
        self.top_k = top_k
        self.return_source_documents = return_source_documents
        self.threshold = similarity_score_threshold
        self.reranking = reranking
        self.rerank_topk = rerank_topk

    @time_it
    def __call__(self, question):
        contexts, _ = similarity_search(self.vector_db, question, self.top_k, self.threshold, self.reranking, rerank_topk=self.rerank_topk)
        input_prompt = get_qa_prompt(self.prompt_template, question, contexts)

        res = self.llm_model.invoke(input_prompt)
        output = {'query': question, 'result': res}
        if self.return_source_documents:
            output['source_documents'] = contexts
        return output


class QAChainHF(QAChain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @time_it
    def __call__(self, question):
        contexts, _ = similarity_search(self.vector_db, question, self.top_k, self.threshold, self.reranking, rerank_topk=self.rerank_topk)
        input_prompt = get_qa_prompt(self.prompt_template, question, contexts)

        res = self.llm_model(input_prompt)
        output = {'query': question, 'result': res[0]["generated_text"]}
        if self.return_source_documents:
            output['source_documents'] = contexts
        return output


class QAChainCPP():
    def __init__(
        self,
        llm_model,
        vector_db,
        prompt_template: PromptTemplate,
        top_k: int = 10,
        return_source_documents: bool = False,
        similarity_score_threshold: Optional[float] = None,
        reranking: bool = False,
        rerank_topk: int = 5,
        model_kwargs: Dict = {"echo": False},
    ):
        self.llm_model = llm_model
        self.vector_db = vector_db
        self.prompt_template = prompt_template
        self.top_k = top_k
        self.return_source_documents = return_source_documents
        self.threshold = similarity_score_threshold
        self.reranking = reranking
        self.rerank_topk = rerank_topk
        self.model_kwargs = model_kwargs

    @time_it
    def __call__(self, question):
        contexts, _ = similarity_search(self.vector_db, question, self.top_k, self.threshold, self.reranking, rerank_topk=self.rerank_topk)
        input_prompt = get_qa_prompt(self.prompt_template, question, contexts)

        res = self.llm_model(input_prompt, **self.model_kwargs)
        output = {'query': question, 'result': res["choices"][0]["text"]}
        if self.return_source_documents:
            output['source_documents'] = contexts
        return output
