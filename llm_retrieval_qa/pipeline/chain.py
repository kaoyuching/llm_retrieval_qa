from typing import Optional, Dict
from langchain.prompts import PromptTemplate

from llm_retrieval_qa.timeit import time_it


def get_qa_prompt(prompt_template, question, contexts):
    if len(contexts) == 0:
        contexts = "Found nothing in the documents"
    else:
        contexts = '\n'.join(contexts)
    input_prompt = prompt_template.format(context=contexts, question=question)
    return input_prompt
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
    ):
        self.llm_model = llm_model
        self.vector_db = vector_db
        self.prompt_template = prompt_template
        self.top_k = top_k
        self.return_source_documents = return_source_documents
        self.threshold = similarity_score_threshold

    @time_it
    def __call__(self, question):
        doc_str, contexts, scores = similarity_search(self.vector_db, question, self.top_k, self.threshold)
        input_prompt = get_qa_prompt(self.prompt_template, question, contexts)

        res = self.llm_model.invoke(input_prompt)
        output = {'query': question, 'result': res}
        if self.return_source_documents:
            output['source_documents'] = doc_str
        return output


class QAChainHF(QAChain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @time_it
    def __call__(self, question):
        doc_str, contexts, scores = similarity_search(self.vector_db, question, self.top_k, self.threshold)
        input_prompt = get_qa_prompt(self.prompt_template, question, contexts)

        res = self.llm_model(input_prompt)
        output = {'query': question, 'result': res[0]["generated_text"]}
        if self.return_source_documents:
            output['source_documents'] = doc_str
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
        model_kwargs: Dict = {"echo": False},
    ):
        self.llm_model = llm_model
        self.vector_db = vector_db
        self.prompt_template = prompt_template
        self.top_k = top_k
        self.return_source_documents = return_source_documents
        self.threshold = similarity_score_threshold
        self.model_kwargs = model_kwargs

    @time_it
    def __call__(self, question):
        doc_str, contexts, scores = similarity_search(self.vector_db, question, self.top_k, self.threshold)
        input_prompt = get_qa_prompt(self.prompt_template, question, contexts)

        res = self.llm_model(input_prompt, **self.model_kwargs)
        output = {'query': question, 'result': res["choices"][0]["text"]}
        if self.return_source_documents:
            output['source_documents'] = doc_str
        return output
